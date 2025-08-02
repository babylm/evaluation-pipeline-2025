# File: data_utils.py
# -------------------
from __future__ import annotations

from transformers import AutoProcessor
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse
import json
from typing import TYPE_CHECKING

from evaluation_pipeline.sentence_zero_shot.read_files import read_files

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.processing_utils import ProcessorMixin


class CompletionRankingDataset(Dataset):

    def __init__(self: CompletionRankingDataset, args: argparse.Namespace):
        self.backend: str = args.backend
        self.processor: ProcessorMixin = AutoProcessor.from_pretrained(args.model_path_or_name, padding_side="right", revision=args.revision_name, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor

        if self.tokenizer.pad_token_id is None:
            if self.backend == "causal":
                self.tokenizer.pad_token_id: int = self.tokenizer.eos_token_id
            else:
                self.tokenizer.pad_token_id = self.tokenizer.cls_token_id

        self.image_token = None
        if args.image_template is not None:
            with open("evaluation_pipeline/templates/image_template.json", "r") as image_template_file:
                image_templates = json.load(image_template_file)
                self.image_template = image_templates[args.image_template]
            self.image_token = self.tokenizer.image_token

        # Load and process the data
        self.data: list[dict[str, str | int | list[str] | list[None] | Image]] = read_files(args)

    def __len__(self: CompletionRankingDataset):
        return len(self.data)

    def process_causal_sentences(self: CompletionRankingDataset, sentence_dict: dict[str, list[str] | list[None]], image: Image | None):
        """Helper function for processing the dictionary associated with an individual
        datapoint for inference with a causal LM.

        Args:
            sentence_dict (dict[str, Any]): The dictionary associated with the datapoint
        """
        sentences = sentence_dict["sentences"]
        completions = sentence_dict["completions"]

        processed_sentence_dict = {}
        for sentence_idx, (sentence, completion) in enumerate(zip(sentences, completions)):
            # Basic outputs
            if image is None:
                tokenizer_output = self.processor(text=sentence, return_offsets_mapping=True)
                sentence_tokens = tokenizer_output["input_ids"]
            else:
                if self.image_token is not None:
                    image_sentence = self.image_template.format(image_token=self.image_token, text=sentence)
                else:
                    image_sentence = sentence
                tokenizer_output = self.processor(text=image_sentence, images=image, return_offsets_mapping=True)
                sentence_tokens = self.processor(text=sentence, return_offsets_mapping=True)["input_ids"]
            tokens = tokenizer_output["input_ids"]
            attention_mask = tokenizer_output["attention_mask"]
            offset_mapping = tokenizer_output['offset_mapping']
            embed_image = torch.FloatTensor(tokenizer_output["pixel_values"]) if image is not None else None
            if len(tokens) == 1 and len(sentence) != 0:
                if sentence_tokens:
                    sentence_tokens = sentence_tokens[0]
                tokens = tokens[0]
                attention_mask = attention_mask[0]
                offset_mapping = offset_mapping[0]

            # Phrase mask (to determine the exact tokens associated with the completion/suffix)
            start_idx = len(tokens) - len(sentence_tokens)
            start_char_idx = len(sentence) - len(completion) + offset_mapping[start_idx][0]
            phrase_indices = []
            for i, (start, end) in enumerate(offset_mapping[start_idx:]):
                # If token overlaps with our phrase's character span
                if end > start_char_idx:
                    phrase_indices.append(i+start_idx)

            phrase_mask = [0 for _ in range(len(tokens))]
            for token_idx in phrase_indices:
                phrase_mask[token_idx] = 1

            processed_sentence_dict[f'sentence_{sentence_idx}_tokens'] = torch.LongTensor(tokens)
            processed_sentence_dict[f'sentence_{sentence_idx}_attn_mask'] = torch.LongTensor(attention_mask)
            processed_sentence_dict[f'sentence_{sentence_idx}_phrase_mask'] = torch.LongTensor(phrase_mask)
            processed_sentence_dict[f'sentence_{sentence_idx}_image'] = embed_image

        return processed_sentence_dict

    def process_mlm_sentences(self, sentence_dict: dict[str, list[str] | list[None]], image: Image | None):
        """Helper function for processing the dictionary associated with an individual
        datapoint for inference with an LM trained with the masked language modeling loss.

        Args:
            sentence_dict (dict[str, Any]): The dictionary associated with the datapoint
        """
        sentences = sentence_dict["sentences"]
        completions = sentence_dict["completions"]
        mask_index = self.tokenizer.mask_token_id

        processed_sentence_dict = {}
        for sentence_idx, (sentence, completion) in enumerate(zip(sentences, completions)):
            # Basic outputs
            if image is None:
                tokenizer_output = self.processor(text=sentence, return_offsets_mapping=True)
            else:
                tokenizer_output = self.processor(text=sentence, images=image, return_offsets_mapping=True)
            tokens = tokenizer_output["input_ids"]
            attention_mask = tokenizer_output["attention_mask"]
            embed_image = torch.FloatTensor(tokenizer_output["pixel_values"]) if image is not None else None

            # Get target tokens
            start_char_idx = len(sentence) - len(completion)
            phrase_indices = []
            target_tokens = []
            for i, (start, end) in enumerate(tokenizer_output['offset_mapping']):
                # If token overlaps with our phrase's character span
                if end > start_char_idx:
                    phrase_indices.append(i)
                    target_tokens.append(tokens[i])

            # Produce masked inputs
            processed_tokens = []
            processed_attention_masks = []
            for mask_replacement_index in phrase_indices:
                curr_tokens = torch.LongTensor(tokens)
                curr_tokens[mask_replacement_index] = mask_index
                processed_tokens.append(curr_tokens)

                curr_attention_mask = torch.LongTensor(attention_mask)
                processed_attention_masks.append(curr_attention_mask)

            processed_sentence_dict[f'sentence_{sentence_idx}_tokens'] = processed_tokens
            processed_sentence_dict[f'sentence_{sentence_idx}_attn_mask'] = processed_attention_masks
            processed_sentence_dict[f'sentence_{sentence_idx}_indices'] = torch.LongTensor(phrase_indices)
            processed_sentence_dict[f'sentence_{sentence_idx}_targets'] = torch.LongTensor(target_tokens)
            processed_sentence_dict[f'sentence_{sentence_idx}_image'] = embed_image

        return processed_sentence_dict

    def process_mntp_sentences(self, sentence_dict: dict[str, list[str] | list[None]], image: Image | None):
        """Helper function for processing the dictionary associated with an individual
        datapoint for inference with an LM trained with masked next token prediction..

        Args:
            sentence_dict (dict[str, Any]): The dictionary associated with the datapoint
        """
        sentences = sentence_dict["sentences"]
        completions = sentence_dict["completions"]
        mask_index = self.tokenizer.mask_token_id
        if self.tokenizer.cls_token_id is not None or self.tokenizer.bos_token_id is not None:
            prepend = 1
        else:
            prepend = 0

        processed_sentence_dict = {}
        for sentence_idx, (sentence, completion) in enumerate(zip(sentences, completions)):
            # Basic outputs
            if image is None:
                tokenizer_output = self.processor(text=sentence, return_offsets_mapping=True)
            else:
                tokenizer_output = self.processor(text=sentence, images=image, return_offsets_mapping=True)
            tokens = tokenizer_output["input_ids"]
            attention_mask = tokenizer_output["attention_mask"]
            embed_image = torch.FloatTensor(tokenizer_output["pixel_values"]) if image is not None else None

            # Get target tokens
            start_char_idx = len(sentence) - len(completion)
            phrase_indices = []
            target_tokens = []
            for i, (start, end) in enumerate(tokenizer_output['offset_mapping']):
                # If token overlaps with our phrase's character span
                if end > start_char_idx and (i != 0 or prepend != 0):
                    phrase_indices.append(i)
                    target_tokens.append(tokens[i])

            # Produce masked inputs
            processed_tokens = []
            processed_attention_masks = []
            for phrase_index in phrase_indices:
                curr_tokens = torch.LongTensor(tokens)
                curr_tokens[phrase_index] = mask_index
                processed_tokens.append(curr_tokens)

                curr_attention_mask = torch.LongTensor(attention_mask)
                processed_attention_masks.append(curr_attention_mask)
            processed_tokens = torch.stack(processed_tokens, dim=0)
            processed_attention_masks = torch.stack(processed_attention_masks, dim=0)

            processed_sentence_dict[f'sentence_{sentence_idx}_tokens'] = processed_tokens
            processed_sentence_dict[f'sentence_{sentence_idx}_attn_mask'] = processed_attention_masks
            processed_sentence_dict[f'sentence_{sentence_idx}_indices'] = torch.LongTensor(phrase_indices) - 1
            processed_sentence_dict[f'sentence_{sentence_idx}_targets'] = torch.LongTensor(target_tokens)
            processed_sentence_dict[f'sentence_{sentence_idx}_image'] = embed_image

        return processed_sentence_dict

    def process_enc_dec_mask_sentences(self, sentence_dict: dict[str, list[str] | list[None]], image: Image | None):
        """Helper function for processing the dictionary associated with an individual
        datapoint for inference with an LM trained with masked next token prediction..

        Args:
            sentence_dict (dict[str, Any]): The dictionary associated with the datapoint
        """
        sentences = sentence_dict["sentences"]
        completions = sentence_dict["completions"]
        if self.tokenizer.mask_token_id is None:
            if self.tokenizer.additional_special_tokens is not None:
                mask_index = self.tokenizer.additional_special_tokens_id[0]
            else:
                raise "Unknown mask token, please specify it in the tokenizer!"
        else:
            mask_index = self.tokenizer.mask_token_id

        if self.tokenizer.bos_token_id is not None:
            dec_token = [self.tokenizer.bos_token_id, mask_index]
        else:
            if self.tokenizer.additional_special_tokens is not None:
                dec_token = [self.tokenizer.additional_special_tokens_id[0]]
            else:
                raise "Unknown BOS token, please specify it in the tokenizer!"

        processed_sentence_dict = {}
        for sentence_idx, (sentence, completion) in enumerate(zip(sentences, completions)):
            # Basic outputs
            tokenizer_output = self.tokenizer(sentence, return_offsets_mapping=True)
            tokens = tokenizer_output["input_ids"]
            attention_mask = tokenizer_output["attention_mask"]

            # Get target tokens
            start_char_idx = len(sentence) - len(completion)
            phrase_indices = []
            target_tokens = []
            for i, (start, end) in enumerate(tokenizer_output['offset_mapping']):
                # If token overlaps with our phrase's character span
                if end > start_char_idx:
                    phrase_indices.append(i)
                    target_tokens.append(tokens[i])

            # Produce masked inputs
            processed_tokens = []
            processed_attention_masks = []
            dec_tokens = []
            dec_att = []
            for phrase_index in phrase_indices:
                curr_tokens = torch.LongTensor(tokens)
                curr_tokens[phrase_index] = mask_index
                processed_tokens.append(curr_tokens)

                curr_attention_mask = torch.LongTensor(attention_mask)
                processed_attention_masks.append(curr_attention_mask)
                dec_tokens.append(torch.LongTensor(dec_token))
                dec_att.append(torch.ones(len(dec_token), dtype=torch.long))
            processed_tokens = torch.stack(processed_tokens, dim=0)
            processed_attention_masks = torch.stack(processed_attention_masks, dim=0)

            processed_sentence_dict[f'sentence_{sentence_idx}_enc_tokens'] = processed_tokens
            processed_sentence_dict[f'sentence_{sentence_idx}_enc_attn_mask'] = processed_attention_masks
            processed_sentence_dict[f'sentence_{sentence_idx}_dec_tokens'] = dec_tokens
            processed_sentence_dict[f'sentence_{sentence_idx}_dec_attn_mask'] = dec_att
            processed_sentence_dict[f'sentence_{sentence_idx}_targets'] = torch.LongTensor(target_tokens)

        return processed_sentence_dict

    def process_enc_dec_prefix_sentences(self, sentence_dict: dict[str, list[str] | list[None]], image: Image | None):
        """Helper function for processing the dictionary associated with an individual
        datapoint for inference with an LM trained with masked next token prediction..

        Args:
            sentence_dict (dict[str, Any]): The dictionary associated with the datapoint
        """
        sentences = sentence_dict["sentences"]
        prefixes = sentence_dict["prefixes"]
        completions = sentence_dict["completions"]
        if self.tokenizer.mask_token_id is None:
            if self.tokenizer.additional_special_tokens is not None:
                mask_index = [self.tokenizer.additional_special_tokens_id[0]]
            else:
                raise "Unknown mask token, please specify it in the tokenizer!"
        else:
            mask_index = [self.tokenizer.mask_token_id]

        if self.tokenizer.cls_token_id is not None:
            cls_index = [self.tokenizer.cls_token_id]
            att_prepend = [1]
        else:
            cls_index = []
            att_prepend = []
        if self.tokenizer.bos_token_id is not None:
            bos_index = [self.tokenizer.bos_token_id]
        else:
            if self.tokenizer.additional_special_tokens is not None:
                bos_index = [self.tokenizer.additional_special_tokens_id[0]]
            else:
                raise "Unknown BOS token, please specify it in the tokenizer!"
        if self.tokenizer.eos_token_id is not None:
            eos_index = [self.tokenizer.eos_token_id]
            att_append = [1]
        else:
            eos_index = []
            att_append = []

        processed_sentence_dict = {}
        for sentence_idx, (sentence, prefix, completion) in enumerate(zip(sentences, prefixes, completions)):
            # Basic outputs
            enc_tokenizer_output = self.tokenizer(prefix, add_special_tokens=False) if prefix is not None else []
            dec_tokenizer_output = self.tokenizer(completion, add_special_tokens=False)
            if enc_tokenizer_output:
                enc_tokens = enc_tokenizer_output["input_ids"]
                enc_attention_mask = enc_tokenizer_output["attention_mask"]
            else:
                enc_tokens = []
                enc_attention_mask = []
            dec_tokens = dec_tokenizer_output["input_ids"]
            dec_attention_mask = dec_tokenizer_output["attention_mask"]

            target_tokens = [token for token in dec_tokens]
            phrase_mask = torch.ones(len(target_tokens), dtype=torch.long)
            processed_sentence_dict[f'sentence_{sentence_idx}_phrase_mask'] = phrase_mask

            processed_tokens = torch.LongTensor(cls_index + enc_tokens + mask_index + eos_index)
            processed_attention_mask = torch.LongTensor(att_prepend + enc_attention_mask + [1] + att_append)

            dec_tokens = torch.LongTensor(bos_index + mask_index + dec_tokens[:-1])
            dec_attention_mask = torch.LongTensor([1, 1] + dec_attention_mask[:-1])

            processed_sentence_dict[f'sentence_{sentence_idx}_enc_tokens'] = processed_tokens
            processed_sentence_dict[f'sentence_{sentence_idx}_enc_attn_mask'] = processed_attention_mask
            processed_sentence_dict[f'sentence_{sentence_idx}_dec_tokens'] = dec_tokens
            processed_sentence_dict[f'sentence_{sentence_idx}_dec_attn_mask'] = dec_attention_mask
            processed_sentence_dict[f'sentence_{sentence_idx}_targets'] = torch.LongTensor(target_tokens)

        return processed_sentence_dict

    def __getitem__(self: CompletionRankingDataset, idx: int):
        data_dict: dict[str, str | int | list[str] | list[None] | Image] = self.data[idx]
        sentence_dict: dict[str, list[str] | list[None]] = {"sentences" : data_dict["sentences"], "prefixes": data_dict["prefixes"], "completions" : data_dict["completions"]}
        label: int = data_dict["label"]
        uid: str = data_dict["UID"]
        if "image" in data_dict.keys():
            image: Image | None = data_dict["image"]
        else:
            image = None

        metadata_keys: list[str] = [key for key in data_dict if key not in ["sentences", "completions", "prefixes", "label", "image"]]
        metadata: dict[str, str] = {key : data_dict[key] for key in metadata_keys}

        if self.backend == "causal":
            processed_sentence_dict: dict[str, torch.Tensor | None] = self.process_causal_sentences(sentence_dict, image)
        elif self.backend == "mlm":
            processed_sentence_dict = self.process_mlm_sentences(sentence_dict, image)
        elif self.backend == "mntp":
            processed_sentence_dict = self.process_mntp_sentences(sentence_dict, image)
        elif self.backend == "enc_dec_mask":
            processed_sentence_dict = self.process_enc_dec_mask_sentences(sentence_dict, image)
        elif self.backend == "enc_dec_prefix":
            processed_sentence_dict = self.process_enc_dec_prefix_sentences(sentence_dict, image)

        return sentence_dict, processed_sentence_dict, label, metadata, uid

    def collate_fn(self: CompletionRankingDataset, batch: tuple[dict[str, list[str] | list[None]], int, dict[str, str], str, Image | None]):
        """The collate function of the dataset, which creates a
        tensor batch of the a batch of dictionaries containing
        all necessary information to evaluate the data.

        Args:
            batch: A tuple of size batch size, where each
                item with 5 elements; a dict with 3 list of
                strings (or None) representing the prefixes,
                the completions, and the sentences; an int
                representing the label; a dict of strings
                containing all the aggregation categories for
                the results; a string representing the UID; an
                Image (or None) representing the image
                associated with the text.

        Returns:
        """
        pass


def get_collate_fn(args: argparse.ArgumentParser, pad_idx: int):
    """Helper function to construct the collation function for evaluation. The collation function
    for causal LMs is distinct from those used for MLM and MNTP backends.

    Args:
        args (argparse.ArgumentParser): Arguments to determine model backend
        pad_idx (int): What token to use as the padding index
    """

    if args.backend == "causal":
        return get_causal_collate_fn(pad_idx)
    elif args.backend in ["mlm", "mntp"]:
        return get_mlm_collate_fn(pad_idx)
    elif args.backend == "enc_dec_mask":
        return get_enc_dec_mask_collate_fn(pad_idx)
    elif args.backend == "enc_dec_prefix":
        return get_enc_dec_prefix_collate_fn(pad_idx)


def get_causal_collate_fn(pad_idx):
    def collate_fn(batch):
        # First pad the tensors
        num_sentences = len([key for key in batch[0][1].keys() if key.endswith("tokens")])
        sentence_dict_with_padding = {}
        for sentence_idx in range(num_sentences):
            # Tokens
            tokens = [item[1][f'sentence_{sentence_idx}_tokens'] for item in batch]
            padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=pad_idx)
            sentence_dict_with_padding[f'sentence_{sentence_idx}_inputs'] = padded_tokens[:, :-1]
            sentence_dict_with_padding[f'sentence_{sentence_idx}_targets'] = padded_tokens[:, 1:]

            # Attention mask
            attention_masks = [item[1][f'sentence_{sentence_idx}_attn_mask'] for item in batch]
            sentence_dict_with_padding[f'sentence_{sentence_idx}_attn_mask'] = pad_sequence(attention_masks, batch_first=True, padding_value=0)[:, :-1]

            # Phrase mask
            phrase_masks = [item[1][f'sentence_{sentence_idx}_phrase_mask'] for item in batch]
            sentence_dict_with_padding[f'sentence_{sentence_idx}_phrase_mask'] = pad_sequence(phrase_masks, batch_first=True, padding_value=0)[:, 1:]

            # Images
            images = [item[1][f'sentence_{sentence_idx}_image'] for item in batch]
            if all(image is None for image in images):
                images = None
            else:
                images = torch.cat(images, dim=0)

        # Next handle the labels and metadata
        sentence_dict = [item[0] for item in batch]
        labels = [item[2] for item in batch]
        metadatas = [item[3] for item in batch]
        uids = [item[4] for item in batch]
        return sentence_dict, sentence_dict_with_padding, labels, metadatas, uids, images
    return collate_fn


def get_mlm_collate_fn(pad_idx):
    def collate_fn(batch):
        # Pad the tensors
        num_sentences = len([key for key in batch[0][1].keys() if key.endswith("tokens")])
        sentence_dict_with_padding = {}
        for sentence_idx in range(num_sentences):
            # Tokens and attention masks
            tokens = []
            attention_masks = []
            examples_per_batch = []
            for item in batch:
                tokens += item[1][f'sentence_{sentence_idx}_tokens']
                attention_masks += item[1][f'sentence_{sentence_idx}_attn_mask']
                examples_per_batch.append(len(item[1][f'sentence_{sentence_idx}_tokens']))

            padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=pad_idx)
            sentence_dict_with_padding[f'sentence_{sentence_idx}_tokens'] = padded_tokens
            padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
            sentence_dict_with_padding[f'sentence_{sentence_idx}_attn_mask'] = padded_attention_masks
            sentence_dict_with_padding[f'sentence_{sentence_idx}_examples_per_batch'] = examples_per_batch

            # Mask indices and targets
            mask_indices = torch.cat([item[1][f'sentence_{sentence_idx}_indices'] for item in batch], dim=0)
            sentence_dict_with_padding[f'sentence_{sentence_idx}_indices'] = mask_indices
            targets = torch.cat([item[1][f'sentence_{sentence_idx}_targets'] for item in batch], dim=0)
            sentence_dict_with_padding[f'sentence_{sentence_idx}_targets'] = targets

            # Images
            images = [item[1][f'sentence_{sentence_idx}_image'] for item in batch]
            if all(image is None for image in images):
                images = None
            else:
                images = torch.cat(images, dim=0)

        # Handle the labels and metadata
        sentence_dict = [item[0] for item in batch]
        labels = [item[2] for item in batch]
        metadatas = [item[3] for item in batch]
        uids = [item[4] for item in batch]
        return sentence_dict, sentence_dict_with_padding, labels, metadatas, uids, images
    return collate_fn


def get_enc_dec_mask_collate_fn(pad_idx):
    def collate_fn(batch):
        # Pad the tensors
        num_sentences = len([key for key in batch[0][1].keys() if key.endswith("enc_tokens")])
        sentence_dict_with_padding = {}
        for sentence_idx in range(num_sentences):
            # Tokens and attention masks
            enc_tokens = []
            enc_attention_masks = []
            dec_tokens = []
            dec_attention_masks = []
            examples_per_batch = []
            for item in batch:
                enc_tokens += item[1][f'sentence_{sentence_idx}_enc_tokens']
                enc_attention_masks += item[1][f'sentence_{sentence_idx}_enc_attn_mask']
                dec_tokens += item[1][f'sentence_{sentence_idx}_dec_tokens']
                dec_attention_masks += item[1][f'sentence_{sentence_idx}_dec_attn_mask']
                examples_per_batch.append(len(item[1][f'sentence_{sentence_idx}_enc_tokens']))

            padded_tokens = pad_sequence(enc_tokens, batch_first=True, padding_value=pad_idx)
            sentence_dict_with_padding[f'sentence_{sentence_idx}_enc_tokens'] = padded_tokens
            padded_attention_masks = pad_sequence(enc_attention_masks, batch_first=True, padding_value=0)
            sentence_dict_with_padding[f'sentence_{sentence_idx}_enc_attn_mask'] = padded_attention_masks
            sentence_dict_with_padding[f'sentence_{sentence_idx}_dec_tokens'] = pad_sequence(dec_tokens, batch_first=True, padding_value=pad_idx)
            sentence_dict_with_padding[f'sentence_{sentence_idx}_dec_attn_mask'] = pad_sequence(dec_attention_masks, batch_first=True, padding_value=0)
            sentence_dict_with_padding[f'sentence_{sentence_idx}_examples_per_batch'] = examples_per_batch

            # Targets
            targets = torch.cat([item[1][f'sentence_{sentence_idx}_targets'] for item in batch], dim=0)
            sentence_dict_with_padding[f'sentence_{sentence_idx}_targets'] = targets

            # Images
            images = [item[1][f'sentence_{sentence_idx}_image'] for item in batch]
            if all(image is None for image in images):
                images = None
            else:
                images = torch.cat(images, dim=0)

        # Handle the labels and metadata
        sentence_dict = [item[0] for item in batch]
        labels = [item[2] for item in batch]
        metadatas = [item[3] for item in batch]
        uids = [item[4] for item in batch]
        return sentence_dict, sentence_dict_with_padding, labels, metadatas, uids, images
    return collate_fn


def get_enc_dec_prefix_collate_fn(pad_idx):
    def collate_fn(batch):
        # First pad the tensors
        num_sentences = len([key for key in batch[0][1].keys() if key.endswith("dec_tokens")])
        sentence_dict_with_padding = {}
        for sentence_idx in range(num_sentences):
            # Tokens and Targets
            enc_tokens = [item[1][f'sentence_{sentence_idx}_enc_tokens'] for item in batch]
            padded_enc_tokens = pad_sequence(enc_tokens, batch_first=True, padding_value=pad_idx)
            sentence_dict_with_padding[f'sentence_{sentence_idx}_enc_tokens'] = padded_enc_tokens
            dec_tokens = [item[1][f'sentence_{sentence_idx}_dec_tokens'] for item in batch]
            padded_dec_tokens = pad_sequence(dec_tokens, batch_first=True, padding_value=pad_idx)
            sentence_dict_with_padding[f'sentence_{sentence_idx}_dec_tokens'] = padded_dec_tokens
            sentence_dict_with_padding[f'sentence_{sentence_idx}_targets'] = pad_sequence([item[1][f'sentence_{sentence_idx}_targets'] for item in batch], batch_first=True, padding_value=pad_idx)

            # Attention mask
            enc_attention_masks = [item[1][f'sentence_{sentence_idx}_enc_attn_mask'] for item in batch]
            sentence_dict_with_padding[f'sentence_{sentence_idx}_enc_attn_mask'] = pad_sequence(enc_attention_masks, batch_first=True, padding_value=0)
            dec_attention_masks = [item[1][f'sentence_{sentence_idx}_dec_attn_mask'] for item in batch]
            sentence_dict_with_padding[f'sentence_{sentence_idx}_dec_attn_mask'] = pad_sequence(dec_attention_masks, batch_first=True, padding_value=0)

            # Phrase mask
            phrase_masks = [item[1][f'sentence_{sentence_idx}_phrase_mask'] for item in batch]
            sentence_dict_with_padding[f'sentence_{sentence_idx}_phrase_mask'] = pad_sequence(phrase_masks, batch_first=True, padding_value=0)

            # Images
            images = [item[1][f'sentence_{sentence_idx}_image'] for item in batch]
            if all(image is None for image in images):
                images = None
            else:
                images = torch.cat(images, dim=0)

        # Next handle the labels and metadata
        sentence_dict = [item[0] for item in batch]
        labels = [item[2] for item in batch]
        metadatas = [item[3] for item in batch]
        uids = [item[4] for item in batch]
        return sentence_dict, sentence_dict_with_padding, labels, metadatas, uids, images
    return collate_fn


def get_dataloader(args):
    """This function constructs the dataset and associated dataloader with collation functions specialized
    to the model backend.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    dataset = CompletionRankingDataset(args)
    pad_idx = dataset.tokenizer.pad_token_id
    collate_fn = get_collate_fn(args, pad_idx)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader
