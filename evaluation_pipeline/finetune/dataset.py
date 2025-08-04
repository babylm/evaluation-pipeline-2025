from __future__ import annotations

import torch
import json
from typing import TYPE_CHECKING
from functools import partial

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    import pathlib


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_file: pathlib.Path, task: str) -> None:
        """This is the supervised dataset for the finetuning tasks.
        This is optimized for sequence classification and works with
        JSONL files.

        Args:
            input_file(pathlib.Path): The path to the JSONL file
                containing the data.
            task(str): The task that data represents. Used to
                determine how the data is read.
        """
        load = partial(self.load_file, input_file)

        match task:
            case "boolq":
                load(["question"], ["passage"])
            case "cola":
                load(["sentence"])
            case "mnli":
                load(["premise"], ["hypothesis"])
            case "mrpc":
                load(["sentence1"], ["sentence2"])
            case "multirc":
                load(["question", "answer"], ["paragraph"], "Question: {} Answer: {}")
            case "qnli":
                load(["question"], ["sentence"])
            case "qqp":
                load(["question1"], ["question2"])
            case "rte":
                load(["sentence1"], ["sentence2"])
            case "sst2":
                load(["sentence"])
            case "wsc":
                load(["span2_text", "span1_text"], ["text"], "Does \"{}\" refer to \"{}\" in this passage?")
            case _:
                raise ValueError("This is not an implemented task! Please implement it!")

    def load_file(self, input_file: pathlib.Path, A_sentence_keys: list[str], B_sentence_keys: list[str] | None = None, A_template: str | None = None, B_template: str | None = None) -> None:
        """This function loads the data from a JSONL file into
        the Dataset class.

        Args:
            input_file(pathlib.Path): The path to the JSONL file
                containing the data.
            A_sentence_keys(list[str]): The list of keys that
                contain the sentences that are to the left of
                the seperator (when a we have a pairs or inputs).
            B_sentence_keys(list[str] | None): The list of keys
                that contain the sentences that are to the right
                of the seperator. If it is None, then we do not
                have a seperator.
            A_template(str | None): How to template the different
                strings in the A_sentence_keys. If it is None, the
                strings are joined with whitespaces seperating
                them.
            B_template(str | None): How to template the different
                strings in the B_sentence_keys. If it is None, the
                strings are joined with whitespaces seperating
                them.
        """
        self.texts = []
        self.labels = []

        with input_file.open("r") as file:
            for line in file:
                data = json.loads(line)

                A_sentences = [data[key] for key in A_sentence_keys]
                if A_template is not None:
                    A_string = A_template.format(*A_sentences)
                else:
                    A_string = " ".join(A_sentences)

                if B_sentence_keys is not None:
                    B_sentences = [data[key] for key in B_sentence_keys]
                    if B_template is not None:
                        B_string = B_template.format(*B_sentences)
                    else:
                        B_string = " ".join(B_sentences)
                    self.texts.append((A_string, B_string))
                else:
                    self.texts.append(A_string)

                self.labels.append(data["label"])

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        text = self.texts[index]
        label = self.labels[index]

        return text, label

    @staticmethod
    def collate_function(tokenizer: PreTrainedTokenizerBase, causal: bool, max_length: int, data: list[tuple[str, str] | int | str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This functions tokenizes, creates a attention_mask and
        collates a batch of texts/pairs of texts.

        Args:
            tokenizer(PreTrainedTokenizerBase): The tokenizer
                corresponding to the model to finetune.
            causal(bool): Whether or not the model expects causal
                attention masking.
            max_length(int): The maximum sequence length before the
                input is truncated.
            data(list[tuple[str, str] | int | str]): A list
                containing a batch of texts/pair of texts and
                labels.

        Returns:
            torch.Tensor: A tensor of the tokenized inputs.
                Shape: :math:`(B, S)`
            torch.Tensor: A tensor of 1s and 0s representing the
                tokens to attend to.
                Shape: :math:`(B, S)` or :math:`(B, S, S)`
            torch.Tensor: A tensor with the correct label for
                each input.
                Shape: :math:`(B)`
        """
        texts = []
        labels = []

        for text, label in data:
            texts.append(text)
            labels.append(label)

        labels = torch.tensor(labels, dtype=torch.long)
        encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        if causal:
            attention_mask = encodings.attention_mask.unsqueeze(1).repeat(1, encodings.attention_mask.size(-1), 1).tril(diagonal=0)
        else:
            attention_mask = encodings.attention_mask

        return encodings.input_ids, attention_mask, labels


class PredictDataset(torch.utils.data.Dataset):

    def __init__(self, input_file: pathlib.Path, task: str) -> None:
        """This is the prediction dataset for the finetuning tasks.
        This is optimized for sequence classification and works with
        JSONL files.

        Args:
            input_file(pathlib.Path): The path to the JSONL file
                containing the data.
            task(str): The task that data represents. Used to
                determine how the data is read.
        """
        load = partial(self.load_file, input_file)

        match task:
            case "boolq":
                load(["question"], ["passage"])
            case "cola":
                load(["sentence"])
            case "mnli":
                load(["premise"], ["hypothesis"])
            case "mrpc":
                load(["sentence1"], ["sentence2"])
            case "multirc":
                load(["question", "answer"], ["paragraph"], "Question: {} Answer: {}")
            case "qnli":
                load(["question"], ["sentence"])
            case "qqp":
                load(["question1"], ["question2"])
            case "rte":
                load(["sentence1"], ["sentence2"])
            case "sst2":
                load(["sentence"])
            case "wsc":
                load(["span2_text", "span1_text"], ["text"], "Does \"{}\" refer to \"{}\" in this passage?")
            case _:
                raise ValueError("This is not an implemented task! Please implement it!")

    def load_file(self, input_file: pathlib.Path, A_sentence_keys: list[str], B_sentence_keys: list[str] | None = None, A_template: str | None = None, B_template: str | None = None) -> None:
        """This function loads the data from a JSONL file into
        the Dataset class.

        Args:
            input_file(pathlib.Path): The path to the JSONL file
                containing the data.
            A_sentence_keys(list[str]): The list of keys that
                contain the sentences that are to the left of
                the seperator (when a we have a pairs or inputs).
            B_sentence_keys(list[str] | None): The list of keys
                that contain the sentences that are to the right
                of the seperator. If it is None, then we do not
                have a seperator.
            A_template(str | None): How to template the different
                strings in the A_sentence_keys. If it is None, the
                strings are joined with whitespaces seperating
                them.
            B_template(str | None): How to template the different
                strings in the B_sentence_keys. If it is None, the
                strings are joined with whitespaces seperating
                them.
        """
        self.texts = []

        with input_file.open("r") as file:
            for line in file:
                data = json.loads(line)

                A_sentences = [data[key] for key in A_sentence_keys]
                if A_template is not None:
                    A_string = A_template.format(*A_sentences)
                else:
                    A_string = " ".join(A_sentences)

                if B_sentence_keys is not None:
                    B_sentences = [data[key] for key in B_sentence_keys]
                    if B_template is not None:
                        B_string = B_template.format(*B_sentences)
                    else:
                        B_string = " ".join(B_sentences)
                    self.texts.append((A_string, B_string))
                else:
                    self.texts.append(A_string)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> str:
        text = self.texts[index]

        return text

    @staticmethod
    def collate_function(tokenizer: PreTrainedTokenizerBase, causal: bool, max_length: int, data: list[tuple[str, str] | int]) -> tuple[torch.Tensor, torch.Tensor]:
        """This functions tokenizes, creates a attention_mask, and
        collates a batch of texts/pairs of texts.

        Args:
            tokenizer(PreTrainedTokenizerBase): The tokenizer
                corresponding to the model to finetune.
            causal(bool): Whether or not the model expects causal
                attention masking.
            max_length(int): The maximum sequence length before the
                input is truncated.
            data(list[tuple[str, str] | int | str]): A list
                containing a batch of texts/pair.

        Returns:
            torch.Tensor: A tensor of the tokenized inputs.
                Shape: :math:`(B, S)`
            torch.Tensor: A tensor of 1s and 0s representing the
                tokens to attend to.
                Shape: :math:`(B, S)` or :math:`(B, S, S)`
        """
        texts = []

        for text in data:
            texts.append(text)

        encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        if causal:
            attention_mask = encodings.attention_mask.unsqueeze(1).repeat(1, encodings.attention_mask.size(-1), 1).tril(diagonal=0)
        else:
            attention_mask = encodings.attention_mask

        return encodings.input_ids, attention_mask
