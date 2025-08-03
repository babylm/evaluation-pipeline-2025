import math
import random
import typing as t
from collections import Counter, defaultdict
from dataclasses import dataclass

import spacy
from datasets import load_dataset
from spellchecker import SpellChecker


@dataclass
class ContextStats:
    """Statistics for a context."""

    count: int
    frequency: float
    log_freq_per_million: float
    subsequent_words: Counter


class TextPreprocessor:
    """Preprocesses text using spaCy for sentence segmentation."""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
        self.nlp.add_pipe("sentencizer")

    def process_text(self, text: str) -> list[list[str]]:
        """Process text into sentences and tokenize."""
        doc = self.nlp(text)
        return [sentence.text.strip().split() for sentence in doc.sents]

    def load_and_process_dataset(
        self, dataset_name: str, split: str, batch_size: int = 1000
    ):
        """Load and process dataset in batches."""
        dataset = load_dataset(dataset_name, split=split)
        if "text" not in dataset.column_names:
            raise ValueError(
                f"'text' column not found in the dataset. Available columns: {dataset.column_names}"
            )

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]["text"]
            for text in batch:
                yield self.process_text(text)


class NGramStatisticsComputer:
    """Computes n-gram statistics using beginning-of-sentence context."""

    def __init__(self, max_context_size: int | None = None):
        self.max_context_size = max_context_size
        self.context_stats: dict[str, ContextStats] = defaultdict(
            lambda: ContextStats(0, 0.0, float("-inf"), Counter())
        )
        self.word_counts = Counter()
        self.total_windows = 0

    def compute_stats(self, processed_sentences: list[list[str]]):
        """Compute statistics from processed sentences using beginning-of-sentence contexts."""
        for sentence in processed_sentences:
            if len(sentence) <= 1:
                continue

            for i in range(1, len(sentence)):
                next_word = sentence[i]
                context_tokens = sentence[:i]

                if (
                    self.max_context_size is not None
                    and len(context_tokens) > self.max_context_size
                ):
                    context_tokens = context_tokens[-self.max_context_size :]

                context_key = " ".join(context_tokens)

                self.context_stats[context_key].count += 1
                self.context_stats[context_key].subsequent_words[next_word] += 1
                self.word_counts[next_word] += 1
                self.total_windows += 1

    def compute_frequencies(self):
        """Compute frequency statistics for all contexts and words."""
        for stats in self.context_stats.values():
            stats.frequency = stats.count / self.total_windows
            stats.log_freq_per_million = self.compute_log_freq_per_million(
                stats.count, self.total_windows
            )

    @staticmethod
    def compute_log_freq_per_million(count: int, total: int) -> float:
        """Compute log frequency per million occurrences."""
        freq_per_million = (count / total) * 1_000_000
        return math.log10(freq_per_million) if freq_per_million > 0 else float("-inf")

    def get_all_ngram_stats(self):
        """Get comprehensive statistics for all n-grams."""
        total_words = sum(self.word_counts.values())

        return {
            "context_stats": {
                context: {
                    "count": stats.count,
                    "frequency": stats.frequency,
                    "log_freq_per_million": stats.log_freq_per_million,
                    "subsequent_words": {
                        word: {
                            "count": count,
                            "frequency": count / stats.count,
                            "log_freq_per_million": self.compute_log_freq_per_million(
                                count, stats.count
                            ),
                        }
                        for word, count in stats.subsequent_words.items()
                    },
                }
                for context, stats in self.context_stats.items()
            },
            "word_stats": {
                word: {
                    "count": count,
                    "frequency": count / total_words,
                    "log_freq_per_million": self.compute_log_freq_per_million(
                        count, total_words
                    ),
                }
                for word, count in self.word_counts.items()
            },
            "metadata": {
                "corpus_stats": {
                    "total_windows": self.total_windows,
                    "total_words": total_words,
                    "unique_contexts": len(self.context_stats),
                    "unique_words": len(self.word_counts),
                    "max_context_size": self.max_context_size,
                }
            },
        }


class NGramContextCollector:
    """Collects and processes n-gram statistics from datasets."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.computer = NGramStatisticsComputer()

    def collect_stats(self, dataset_name: str, split: str, batch_size: int = 1000):
        """Collect statistics from the specified dataset."""
        for processed_batch in self.preprocessor.load_and_process_dataset(
            dataset_name, split, batch_size
        ):
            self.computer.compute_stats(processed_batch)

        self.computer.compute_frequencies()

    def get_all_ngram_stats(self) -> dict[str, t.Any]:
        """Get comprehensive n-gram statistics."""
        return self.computer.get_all_ngram_stats()

    @staticmethod
    def filter_contexts(
        ngram_stats: dict[str, t.Any],
        target_words: list[str],
        n_contexts: int,
        mode: str = "frequent",
        min_window_size: int = 1,
    ) -> dict[str, list[dict[str, t.Any]]]:
        """Filter and select contexts for target words based on specified criteria."""
        selected_data: dict[str, list[dict[str, t.Any]]] = {}

        for word in target_words:
            word_contexts = []

            for context_key, stats in ngram_stats["context_stats"].items():
                context_tokens = context_key.split()
                if len(context_tokens) < min_window_size:
                    continue

                if word in stats["subsequent_words"]:
                    word_count = stats["subsequent_words"][word]["count"]
                    word_freq = stats["subsequent_words"][word]["frequency"]
                    word_log_freq = stats["subsequent_words"][word][
                        "log_freq_per_million"
                    ]

                    word_contexts.append(
                        {
                            "context": context_key,
                            "word_in_context_stats": {
                                "count": word_count,
                                "frequency": word_freq,
                                "log_freq_per_million": word_log_freq,
                            },
                        }
                    )

            sorted_contexts = sorted(
                word_contexts,
                key=lambda x: x["word_in_context_stats"]["count"],
                reverse=True,
            )
            n_select = min(n_contexts, len(sorted_contexts))

            if n_select == 0:
                selected_data[word] = []
            elif mode == "random" and n_select > 0:
                selected = random.sample(sorted_contexts, n_select)
                selected_data[word] = selected
            else:
                selected = sorted_contexts[:n_select]
                selected_data[word] = selected

        return selected_data


def is_word(word: str) -> bool:
    """Check if a word is correctly spelled."""
    spell = SpellChecker()
    return word.lower() in spell


def filter_words(word_list: list[str]) -> list[str]:
    """Filter a list and return only correctly spelled words."""
    spell = SpellChecker()
    return [word for word in word_list if word.lower() in spell]
