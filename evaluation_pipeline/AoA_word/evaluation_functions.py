#!/usr/bin/env python
import logging
import random
import typing as t
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
)

from evaluation_pipeline.AoA_word.eval_util import JsonProcessor, StepConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
random.seed(42)
T = t.TypeVar("T")


class StepSurprisalExtractor:
    """Extracts word surprisal across different training steps."""

    def __init__(
        self,
        config: StepConfig,
        model_name: str,
        backend: str,
        device: str,
        model_cache_dir: Path = None,
    ) -> None:
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.backend = backend
        self.config = config
        self.device = device
        self.current_step = None
        logger.info(f"Using device: {self.device}")
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration."""
        if isinstance(self.model_cache_dir, str):
            self.model_cache_dir = Path(self.model_cache_dir)

    def load_model_for_step(self, step: int) -> AutoModelForCausalLM:
        """Load model and tokenizer for a specific step."""
        try:
            if self.backend in ["mlm", "mntp"]:
                model = AutoModelForMaskedLM.from_pretrained(
                    self.model_name, trust_remote_code=True, revision=step
                )
            elif self.backend == "causal":
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, trust_remote_code=True, revision=step
                )
            elif self.backend in ["enc_dec_mask", "enc_dec_prefix"]:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name, trust_remote_code=True, revision=step
                )
            else:
                raise f"The backend {self.backend} is not implemented, please implemented yourself or raise an issue on the GitHub!"
            model = model.to(self.device)
            model.eval()
        except Exception as e:
            logger.error(f"Error loading model for step {step}: {e!s}")
            raise
        return model

    def load_tokenizer_for_step(self, step: int) -> AutoProcessor:
        try:
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="right",
                revision=step,
            )
            tokenizer = (
                processor.tokenizer if hasattr(processor, "tokenizer") else processor
            )
        except Exception as e:
            logger.error(f"Error loading tokenizer for step {step}: {e!s}")
            raise
        return processor, tokenizer

    def compute_surprisal(
        self,
        model: AutoModelForCausalLM,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        context: str,
        target_word: str,
        use_bos_only: bool = True,
    ) -> float:
        context = context.strip() + " "
        target_word = target_word.strip()
        try:
            if self.backend == "causal":
                return self.compute_surprisal_causal(
                    model, processor, tokenizer, context, target_word, use_bos_only
                )
            elif self.backend == "mntp":
                return self.compute_surprisal_mntp(
                    model, processor, tokenizer, context, target_word, use_bos_only
                )
            elif self.backend == "mlm":
                return self.compute_surprisal_mlm(
                    model, processor, tokenizer, context, target_word, use_bos_only
                )
            elif self.backend == "enc_dec_mask":
                return self.compute_surprisal_enc_dec_mask(
                    model, processor, tokenizer, context, target_word, use_bos_only
                )
            elif self.backend == "enc_dec_prefix":
                return self.compute_surprisal_enc_dec_prefix(
                    model, processor, tokenizer, context, target_word, use_bos_only
                )
            else:
                raise "Unknown backend!"
        except Exception as e:
            logger.error(f"Error in compute surprisal: {e!s}")
            return float("nan")

    def compute_surprisal_causal(
        self,
        model: AutoModelForCausalLM,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        context: str,
        target_word: str,
        use_bos_only: bool = True,
    ) -> float:
        input_tokens, target_tokens, attn_mask, phrase_mask = self.process_causal_input(
            processor, tokenizer, context, target_word, use_bos_only
        )
        with torch.no_grad():
            logits = model(input_ids=input_tokens, attention_mask=attn_mask)
            if isinstance(logits, tuple):
                logits = logits[0]  # BxTxV
            else:
                logits = logits["logits"]  # BxTxV

            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = torch.gather(
                log_probs, -1, target_tokens.unsqueeze(-1)
            ).squeeze(-1)
            phrase_log_probs = torch.sum(target_log_probs * phrase_mask, dim=1)
            surprisal = -phrase_log_probs[0].item()
        return surprisal

    def process_causal_input(
        self,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        context: str,
        target_word: str,
        use_bos_only: bool = True,
    ):
        # Get the input text
        if use_bos_only:
            bos_token = tokenizer.bos_token
            input_text = bos_token + target_word
        else:
            input_text = context + target_word

        # Tokenize overall context
        tokenizer_output = processor(text=input_text, return_offsets_mapping=True)
        start_char_idx = len(input_text) - len(target_word)
        offset_mapping = tokenizer_output["offset_mapping"]

        # Determine what indices the target word occupies
        phrase_indices = []
        for i, (start, end) in enumerate(offset_mapping):
            if end > start_char_idx:
                phrase_indices.append(i)
        phrase_mask = [0 for _ in range(len(tokenizer_output["input_ids"]))]
        for token_idx in phrase_indices:
            phrase_mask[token_idx] = 1

        tokens = (
            torch.LongTensor(tokenizer_output["input_ids"]).to(self.device).unsqueeze(0)
        )
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        attention_mask = (
            torch.LongTensor(tokenizer_output["attention_mask"])
            .to(self.device)
            .unsqueeze(0)[:, :-1]
        )
        phrase_mask = torch.LongTensor(phrase_mask).to(self.device).unsqueeze(0)[:, 1:]

        return input_tokens, target_tokens, attention_mask, phrase_mask

    def compute_surprisal_mntp(
        self,
        model: AutoModelForCausalLM,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        context: str,
        target_word: str,
        use_bos_only: bool = True,
    ) -> float:
        tokens, attn_masks, indices, targets = self.process_mntp_input(
            processor, tokenizer, context, target_word, use_bos_only
        )
        with torch.no_grad():
            logits = model(
                input_ids=tokens,
                attention_mask=attn_masks,
            )
            if isinstance(logits, tuple):
                logits = logits[0]  # BxTxV
            else:
                logits = logits["logits"]  # BxTxV

            minibatch_indices = torch.arange(logits.shape[0]).to(self.device)
            masked_logits = logits[minibatch_indices, indices]  # BxV

            log_probs = F.log_softmax(masked_logits, dim=-1)
            target_log_probs = torch.gather(
                log_probs, -1, targets.unsqueeze(-1)
            ).squeeze(-1)  # B
            surprisal = -target_log_probs.sum().item()
        return surprisal

    def process_mntp_input(
        self,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        context: str,
        target_word: str,
        use_bos_only: bool = True,
    ):
        # Get the input text
        if use_bos_only:
            bos_token = tokenizer.bos_token
            input_text = bos_token + target_word
        else:
            input_text = context + target_word
        mask_index = tokenizer.mask_token_id
        if tokenizer.cls_token_id is not None or tokenizer.bos_token_id is not None:
            prepend = 1
        else:
            prepend = 0

        tokenizer_output = processor(text=input_text, return_offsets_mapping=True)
        tokens = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output["attention_mask"]

        # Get target tokens
        start_char_idx = len(input_text) - len(target_word)
        phrase_indices = []
        target_tokens = []
        for i, (start, end) in enumerate(tokenizer_output["offset_mapping"]):
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
        phrase_indices = torch.LongTensor(phrase_indices) - 1
        target_tokens = torch.LongTensor(target_tokens)

        return (
            processed_tokens.to(self.device),
            processed_attention_masks.to(self.device),
            phrase_indices.to(self.device),
            target_tokens.to(self.device),
        )

    def compute_surprisal_mlm(
        self,
        model: AutoModelForCausalLM,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        context: str,
        target_word: str,
        use_bos_only: bool = True,
    ) -> float:
        tokens, attn_masks, indices, targets = self.process_mlm_input(
            processor, tokenizer, context, target_word, use_bos_only
        )
        with torch.no_grad():
            logits = model(
                input_ids=tokens,
                attention_mask=attn_masks,
            )
            if isinstance(logits, tuple):
                logits = logits[0]  # BxTxV
            else:
                logits = logits["logits"]  # BxTxV

            minibatch_indices = torch.arange(logits.shape[0]).to(self.device)
            masked_logits = logits[minibatch_indices, indices]  # BxV

            log_probs = F.log_softmax(masked_logits, dim=-1)
            target_log_probs = torch.gather(
                log_probs, -1, targets.unsqueeze(-1)
            ).squeeze(-1)  # B
            surprisal = -target_log_probs.sum().item()
        return surprisal

    def process_mlm_input(
        self,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        context: str,
        target_word: str,
        use_bos_only: bool = True,
    ):
        # Get the input text
        if use_bos_only:
            bos_token = tokenizer.bos_token
            input_text = bos_token + target_word
        else:
            input_text = context + target_word
        mask_index = tokenizer.mask_token_id

        tokenizer_output = processor(text=input_text, return_offsets_mapping=True)
        tokens = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output["attention_mask"]

        # Get target tokens
        start_char_idx = len(input_text) - len(target_word)
        phrase_indices = []
        target_tokens = []
        for i, (start, end) in enumerate(tokenizer_output["offset_mapping"]):
            # If token overlaps with our phrase's character span
            if end > start_char_idx:
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
        phrase_indices = torch.LongTensor(phrase_indices)
        target_tokens = torch.LongTensor(target_tokens)

        return (
            processed_tokens.to(self.device),
            processed_attention_masks.to(self.device),
            phrase_indices.to(self.device),
            target_tokens.to(self.device),
        )

    def compute_surprisal_enc_dec_mask(
        self,
        model: AutoModelForCausalLM,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        context: str,
        target_word: str,
        use_bos_only: bool = True,
    ) -> float:
        tokens, attn_mask, dec_input_ids, dec_attn_mask, targets = (
            self.process_enc_dec_mask_input(
                processor, tokenizer, context, target_word, use_bos_only
            )
        )

        with torch.no_grad():
            logits = model(
                input_ids=tokens,
                attention_mask=attn_mask,
                decoder_input_ids=dec_input_ids,
                decoder_attention_mask=dec_attn_mask,
            )
            if isinstance(logits, tuple):
                logits = logits[0]  # BxTxV
            else:
                logits = logits["logits"]  # BxTxV

            masked_logits = logits[:, -1]
            log_probs = F.log_softmax(masked_logits, dim=-1)
            target_log_probs = torch.gather(
                log_probs, -1, targets.unsqueeze(-1)
            ).squeeze(-1)  # B
            surprisal = -target_log_probs.sum().item()

        return surprisal

    def process_enc_dec_mask_input(
        self,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        context: str,
        target_word: str,
        use_bos_only: bool = True,
    ):
        # Get the input text
        if use_bos_only:
            bos_token = tokenizer.bos_token
            input_text = bos_token + target_word
        else:
            input_text = context + target_word

        if tokenizer.mask_token_id is None:
            if tokenizer.additional_special_tokens is not None:
                mask_index = tokenizer.additional_special_tokens_id[0]
            else:
                raise "Unknown mask token, please specify it in the tokenizer!"
        else:
            mask_index = tokenizer.mask_token_id

        if tokenizer.bos_token_id is not None:
            dec_token = [tokenizer.bos_token_id, mask_index]
        else:
            if tokenizer.additional_special_tokens is not None:
                dec_token = [tokenizer.additional_special_tokens_id[0]]
            else:
                raise "Unknown BOS token, please specify it in the tokenizer!"

        tokenizer_output = processor(input_text, return_offsets_mapping=True)
        tokens = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output["attention_mask"]

        # Get target tokens
        start_char_idx = len(input_text) - len(target_word)
        phrase_indices = []
        target_tokens = []
        for i, (start, end) in enumerate(tokenizer_output["offset_mapping"]):
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
        processed_tokens = torch.stack(processed_tokens, dim=0).to(self.device)
        processed_attention_masks = torch.stack(processed_attention_masks, dim=0).to(
            self.device
        )
        dec_tokens = torch.stack(dec_tokens, dim=0).to(self.device)
        dec_att = torch.stack(dec_att, dim=0).to(self.device)
        target_tokens = torch.LongTensor(target_tokens).to(self.device)

        return (
            processed_tokens,
            processed_attention_masks,
            dec_tokens,
            dec_att,
            target_tokens,
        )

    def compute_surprisal_enc_dec_prefix(
        self,
        model: AutoModelForCausalLM,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        context: str,
        target_word: str,
        use_bos_only: bool = True,
    ) -> float:
        input_ids, attn_mask, dec_input_ids, dec_attn_mask, targets, phrase_mask = (
            self.process_enc_dec_prefix_input(
                processor, tokenizer, context, target_word, use_bos_only
            )
        )

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                decoder_input_ids=dec_input_ids,
                decoder_attention_mask=dec_attn_mask,
            )
            if isinstance(logits, tuple):
                logits = logits[0]  # BxTxV
            else:
                logits = logits["logits"]  # BxTxV

            log_probs = F.log_softmax(logits, dim=-1)
            start_pred_token = log_probs.size(1) - targets.size(1)
            target_log_probs = torch.gather(
                log_probs[:, start_pred_token:], -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            phrase_log_probs = torch.sum(target_log_probs * phrase_mask, dim=1)
            surprisal = -phrase_log_probs[0].item()

        return surprisal

    def process_enc_dec_prefix_input(
        self,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        context: str,
        target_word: str,
        use_bos_only: bool = True,
    ):
        # Get the input text
        if use_bos_only:
            bos_token = tokenizer.bos_token
            input_text = bos_token + target_word
        else:
            input_text = context + target_word

        if tokenizer.mask_token_id is None:
            if tokenizer.additional_special_tokens is not None:
                mask_index = [tokenizer.additional_special_tokens_id[0]]
            else:
                raise "Unknown mask token, please specify it in the tokenizer!"
        else:
            mask_index = [tokenizer.mask_token_id]

        if tokenizer.cls_token_id is not None:
            cls_index = [tokenizer.cls_token_id]
            att_prepend = [1]
        else:
            cls_index = []
            att_prepend = []
        if tokenizer.bos_token_id is not None:
            bos_index = [tokenizer.bos_token_id]
        else:
            if tokenizer.additional_special_tokens is not None:
                bos_index = [tokenizer.additional_special_tokens_id[0]]
            else:
                raise "Unknown BOS token, please specify it in the tokenizer!"
        if tokenizer.eos_token_id is not None:
            eos_index = [tokenizer.eos_token_id]
            att_append = [1]
        else:
            eos_index = []
            att_append = []

        enc_tokenizer_output = processor(context, add_special_tokens=False)
        dec_tokenizer_output = processor(target_word, add_special_tokens=False)
        if enc_tokenizer_output:
            enc_tokens = enc_tokenizer_output["input_ids"]
            enc_attention_mask = enc_tokenizer_output["attention_mask"]
        else:
            enc_tokens = []
            enc_attention_mask = []
        dec_tokens = dec_tokenizer_output["input_ids"]
        dec_attention_mask = dec_tokenizer_output["attention_mask"]

        target_tokens = (
            torch.LongTensor([token for token in dec_tokens])
            .to(self.device)
            .unsqueeze(0)
        )
        phrase_mask = (
            torch.ones(len(target_tokens), dtype=torch.long)
            .to(self.device)
            .unsqueeze(0)
        )
        processed_tokens = (
            torch.LongTensor(cls_index + enc_tokens + mask_index + eos_index)
            .to(self.device)
            .unsqueeze(0)
        )
        processed_attention_mask = (
            torch.LongTensor(att_prepend + enc_attention_mask + [1] + att_append)
            .to(self.device)
            .unsqueeze(0)
        )
        dec_tokens = (
            torch.LongTensor(bos_index + mask_index + dec_tokens[:-1])
            .to(self.device)
            .unsqueeze(0)
        )
        dec_attention_mask = (
            torch.LongTensor([1, 1] + dec_attention_mask[:-1])
            .to(self.device)
            .unsqueeze(0)
        )

        return (
            processed_tokens,
            processed_attention_mask,
            dec_tokens,
            dec_attention_mask,
            target_tokens,
            phrase_mask,
        )

    def analyze_steps(
        self,
        contexts: list[list[str]],
        target_words: list[str],
        use_bos_only: bool = False,
        resume_path: Path | None = None,
    ) -> dict[str, t.Any]:
        """Analyze surprisal across steps and return JSON-compatible data."""
        existing_results = []
        if resume_path and resume_path.is_file():
            try:
                existing_data = JsonProcessor.load_json(resume_path)
                if isinstance(existing_data, dict) and "results" in existing_data:
                    existing_results = existing_data["results"]
                elif isinstance(existing_data, list):
                    existing_results = existing_data
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")

        results = []
        for step, word_count in zip(self.config.steps, self.config.word_counts):
            print(f"Checkpoint: {step}")
            try:
                model = self.load_model_for_step(step)
                processor, tokenizer = self.load_tokenizer_for_step(step)

                for word_contexts, target_word in tqdm(
                    zip(contexts, target_words, strict=False)
                ):
                    for context_idx, context in enumerate(word_contexts):
                        surprisal = self.compute_surprisal(
                            model,
                            processor,
                            tokenizer,
                            context,
                            target_word,
                            use_bos_only=use_bos_only,
                        )

                        result_entry = {
                            "step": step,
                            "word_count": word_count,
                            "target_word": target_word,
                            "context_id": context_idx,
                            "context": "BOS_ONLY" if use_bos_only else context,
                            "surprisal": surprisal,
                        }
                        results.append(result_entry)

                if resume_path:
                    interim_data = {
                        "metadata": {
                            "model_name": self.model_name,
                            "use_bos_only": use_bos_only,
                            "total_steps": len(self.config.steps),
                            "completed_steps": len({r["step"] for r in results}),
                        },
                        "results": existing_results + results,
                    }
                    JsonProcessor.save_json(interim_data, resume_path)

                del model, tokenizer
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing step {step}: {e!s}")
                continue

        final_data = {
            "metadata": {
                "model_name": self.model_name,
                "use_bos_only": use_bos_only,
                "total_steps": len(self.config.steps),
                "completed_steps": len({r["step"] for r in results}),
            },
            "results": existing_results + results,
        }

        return final_data
