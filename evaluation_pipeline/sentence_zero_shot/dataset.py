# File: data_utils.py
# -------------------

from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse
from typing import Any

from evaluation_pipeline.sentence_zero_shot.read_files import read_files


class CompletionRankingDataset(Dataset):

    def __init__(self, args: argparse.Namespace):
        self.backend = args.backend
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, padding_side="right", revision=args.revision_name)

        if self.tokenizer.pad_token_id is None:
            if self.backend == "causal":
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.pad_token_id = self.tokenizer.cls_token_id

        # Load and process the data
        self.data = read_files(args.data_path, args.task, args.full_sentence_scores)

    def __len__(self):
        return len(self.data)

    def process_causal_sentences(self, sentence_dict: dict[str, Any]):
        """Helper function for processing the dictionary associated with an individual
        datapoint for inference with a causal LM.

        Args:
            sentence_dict (dict[str, Any]): The dictionary associated with the datapoint
        """
        sentences = sentence_dict["sentences"]
        completions = sentence_dict["completions"]
        bos_index = [self.tokenizer.bos_token_id]

        processed_sentence_dict = {}
        for sentence_idx, (sentence, completion) in enumerate(zip(sentences, completions)):
            # Basic outputs
            tokenizer_output = self.tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)
            tokens = tokenizer_output["input_ids"]
            attention_mask = tokenizer_output["attention_mask"]

            # Phrase mask (to determine the exact tokens associated with the completion/suffix)
            start_char_idx = len(sentence) - len(completion)
            phrase_indices = []
            for i, (start, end) in enumerate(tokenizer_output['offset_mapping']):
                # If token overlaps with our phrase's character span
                if end > start_char_idx:
                    phrase_indices.append(i)

            phrase_mask = [0 for _ in range(len(tokens))]
            for token_idx in phrase_indices:
                phrase_mask[token_idx] = 1

            if bos_index[0] is not None:
                processed_sentence_dict[f'sentence_{sentence_idx}_tokens'] = torch.LongTensor(bos_index + tokens)
                processed_sentence_dict[f'sentence_{sentence_idx}_attn_mask'] = torch.LongTensor([1] + attention_mask)
                processed_sentence_dict[f'sentence_{sentence_idx}_phrase_mask'] = torch.LongTensor([0] + phrase_mask)
            else:
                processed_sentence_dict[f'sentence_{sentence_idx}_tokens'] = torch.LongTensor(tokens)
                processed_sentence_dict[f'sentence_{sentence_idx}_attn_mask'] = torch.LongTensor(attention_mask)
                processed_sentence_dict[f'sentence_{sentence_idx}_phrase_mask'] = torch.LongTensor(phrase_mask)

        return processed_sentence_dict

    def process_mlm_sentences(self, sentence_dict: dict[str, Any]):
        """Helper function for processing the dictionary associated with an individual
        datapoint for inference with an LM trained with the masked language modeling loss.

        Args:
            sentence_dict (dict[str, Any]): The dictionary associated with the datapoint
        """
        sentences = sentence_dict["sentences"]
        completions = sentence_dict["completions"]
        mask_index = self.tokenizer.mask_token_id
        cls_index = [self.tokenizer.cls_token_id]
        sep_index = [self.tokenizer.sep_token_id]

        processed_sentence_dict = {}
        for sentence_idx, (sentence, completion) in enumerate(zip(sentences, completions)):
            # Basic outputs
            tokenizer_output = self.tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)
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
            mask_indices = [phrase_idx + 1 for phrase_idx in phrase_indices]
            for mask_replacement_index in mask_indices:
                curr_tokens = torch.LongTensor(cls_index + tokens + sep_index)
                curr_tokens[mask_replacement_index] = mask_index
                processed_tokens.append(curr_tokens)

                curr_attention_mask = torch.LongTensor([1] + attention_mask + [1])
                processed_attention_masks.append(curr_attention_mask)

            processed_sentence_dict[f'sentence_{sentence_idx}_tokens'] = processed_tokens
            processed_sentence_dict[f'sentence_{sentence_idx}_attn_mask'] = processed_attention_masks
            processed_sentence_dict[f'sentence_{sentence_idx}_indices'] = torch.LongTensor(mask_indices)
            processed_sentence_dict[f'sentence_{sentence_idx}_targets'] = torch.LongTensor(target_tokens)

        return processed_sentence_dict

    def process_mntp_sentences(self, sentence_dict: dict[str, Any]):
        """Helper function for processing the dictionary associated with an individual
        datapoint for inference with an LM trained with masked next token prediction..

        Args:
            sentence_dict (dict[str, Any]): The dictionary associated with the datapoint
        """
        sentences = sentence_dict["sentences"]
        completions = sentence_dict["completions"]
        mask_index = self.tokenizer.mask_token_id
        cls_index = [self.tokenizer.cls_token_id]

        processed_sentence_dict = {}
        for sentence_idx, (sentence, completion) in enumerate(zip(sentences, completions)):
            # Basic outputs
            tokenizer_output = self.tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)
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
            for phrase_index in phrase_indices:
                curr_tokens = torch.LongTensor(cls_index + tokens)
                curr_tokens[phrase_index + 1] = mask_index
                processed_tokens.append(curr_tokens)

                curr_attention_mask = torch.LongTensor([1] + attention_mask)
                processed_attention_masks.append(curr_attention_mask)
            processed_tokens = torch.stack(processed_tokens, dim=0)
            processed_attention_masks = torch.stack(processed_attention_masks, dim=0)

            processed_sentence_dict[f'sentence_{sentence_idx}_tokens'] = processed_tokens
            processed_sentence_dict[f'sentence_{sentence_idx}_attn_mask'] = processed_attention_masks
            processed_sentence_dict[f'sentence_{sentence_idx}_indices'] = torch.LongTensor(phrase_indices)
            processed_sentence_dict[f'sentence_{sentence_idx}_targets'] = torch.LongTensor(target_tokens)

        return processed_sentence_dict

    def __getitem__(self, idx: int):
        data_dict = self.data[idx]
        sentence_dict = {"sentences" : data_dict["sentences"], "completions" : data_dict["completions"]}
        label = data_dict["label"]
        uid = data_dict["UID"]

        metadata_keys = [key for key in data_dict if key not in ["sentences", "completions", "label"]]
        metadata = {key : data_dict[key] for key in metadata_keys}

        if self.backend == "causal":
            processed_sentence_dict = self.process_causal_sentences(sentence_dict)
        elif self.backend == "mlm":
            processed_sentence_dict = self.process_mlm_sentences(sentence_dict)
        else:
            processed_sentence_dict = self.process_mntp_sentences(sentence_dict)

        return sentence_dict, processed_sentence_dict, label, metadata, uid


def get_collate_fn(args: argparse.ArgumentParser, pad_idx: int):
    """Helper function to construct the collation function for evaluation. The collation function
    for causal LMs is distinct from those used for MLM and MNTP backends.

    Args:
        args (argparse.ArgumentParser): Arguments to determine model backend
        pad_idx (int): What token to use as the padding index
    """
    if args.backend == "causal":
        return get_causal_collate_fn(pad_idx)
    else:
        return get_mlm_collate_fn(pad_idx)


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

        # Next handle the labels and metadata
        sentence_dict = [item[0] for item in batch]
        labels = [item[2] for item in batch]
        metadatas = [item[3] for item in batch]
        uids = [item[4] for item in batch]
        return sentence_dict, sentence_dict_with_padding, labels, metadatas, uids
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

        # Handle the labels and metadata
        sentence_dict = [item[0] for item in batch]
        labels = [item[2] for item in batch]
        metadatas = [item[3] for item in batch]
        uids = [item[4] for item in batch]
        return sentence_dict, sentence_dict_with_padding, labels, metadatas, uids
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
