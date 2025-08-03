#!/usr/bin/env python
import logging
import random
import typing as t
from pathlib import Path

import torch
from eval_util import JsonProcessor, StepConfig
from transformers import AutoTokenizer, GPTNeoXForCausalLM

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
        model_cache_dir: Path,
        device: str,
    ) -> None:
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.config = config
        self.device = device
        self.current_step = None
        logger.info(f"Using device: {self.device}")
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration."""
        if isinstance(self.model_cache_dir, str):
            self.model_cache_dir = Path(self.model_cache_dir)

    def load_model_for_step(
        self, step: int
    ) -> tuple[GPTNeoXForCausalLM, AutoTokenizer]:
        """Load model and tokenizer for a specific step."""
        cache_dir = self.model_cache_dir / f"step{step}"

        try:
            model = GPTNeoXForCausalLM.from_pretrained(
                self.model_name, revision=f"step{step}", cache_dir=cache_dir
            )
            model = model.to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=f"step{step}", cache_dir=cache_dir
            )
            model.eval()
            self.current_step = step
            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading model for step {step}: {e!s}")
            raise

    def compute_surprisal(
        self,
        model: GPTNeoXForCausalLM,
        tokenizer: AutoTokenizer,
        context: str,
        target_word: str,
        use_bos_only: bool = True,
    ) -> float:
        """Compute surprisal for a target word given a context."""
        try:
            if use_bos_only:
                bos_token = tokenizer.bos_token
                input_text = bos_token + target_word
                bos_tokens = tokenizer(bos_token, return_tensors="pt").to(self.device)
                context_length = bos_tokens.input_ids.shape[1]
            else:
                input_text = context + target_word
                context_tokens = tokenizer(context, return_tensors="pt").to(self.device)
                context_length = context_tokens.input_ids.shape[1]

            inputs = tokenizer(input_text, return_tensors="pt").to(self.device)

            if context_length >= inputs.input_ids.shape[1]:
                context_length = inputs.input_ids.shape[1] - 1

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                target_logits = logits[0, context_length - 1 : context_length]

                if context_length < inputs.input_ids.shape[1]:
                    target_token_id = inputs.input_ids[0, context_length].item()
                    log_prob = torch.log_softmax(target_logits, dim=-1)[
                        0, target_token_id
                    ]
                    surprisal = -log_prob.item()
                else:
                    logger.error(
                        "Cannot compute surprisal: unable to identify target token"
                    )
                    surprisal = float("nan")

            return surprisal

        except Exception as e:
            logger.error(f"Error in compute_surprisal: {e!s}")
            return float("nan")

    def analyze_steps(
        self,
        contexts: list[list[str]],
        target_words: list[str],
        use_bos_only: bool = True,
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

        for step in self.config.steps:
            try:
                model, tokenizer = self.load_model_for_step(step)

                for word_contexts, target_word in zip(
                    contexts, target_words, strict=False
                ):
                    for context_idx, context in enumerate(word_contexts):
                        surprisal = self.compute_surprisal(
                            model,
                            tokenizer,
                            context,
                            target_word,
                            use_bos_only=use_bos_only,
                        )

                        result_entry = {
                            "step": step,
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
