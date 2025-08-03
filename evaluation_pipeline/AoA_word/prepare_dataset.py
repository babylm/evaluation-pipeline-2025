#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

import pandas as pd
from dataset_util import NGramContextCollector
from eval_util import JsonProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect n-gram contexts from a corpus."
    )
    parser.add_argument(
        "-w",
        "--words_file",
        type=Path,
        default="matched/oxford-understand.csv",
        help="Relative path to the target words",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        default="context",
        help="Relative path to the extracted context",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="stas/c4-en-10k", help="Dataset name"
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument(
        "-s", "--window_size", type=int, default=5, help="Min context window size"
    )
    parser.add_argument(
        "-n", "--n_contexts", type=int, default=20, help="Context numbers"
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["random", "frequent"],
        default="frequent",
        help="Selection mode",
    )
    return parser.parse_args()


def load_target_words(file_path: Path) -> list[str]:
    """Load target words from a file (TXT or CSV format)."""
    if file_path.suffix == ".txt":
        with open(file_path, encoding="utf-8") as f:
            return [line.strip() for line in f]
    elif file_path.suffix == ".csv":
        data = pd.read_csv(file_path)
        logger.info("CSV file has been loaded")
        return data["word"].to_list()
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def main():
    """Main function to collect and process n-gram contexts."""
    args = parse_args()
    words_file = args.words_file
    output_dir = args.output_path / args.dataset / str(args.window_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    ngram_stats_file = output_dir / "ngram_stats.json"
    selected_contexts_file = output_dir / f"{Path(args.words_file).stem}.json"

    logger.info(f"Output file: {selected_contexts_file}")

    target_words = load_target_words(words_file)
    logger.info(f"Loaded {len(target_words)} target words")

    if ngram_stats_file.exists():
        logger.info(f"Loading existing n-gram statistics from {ngram_stats_file}")
        ngram_stats = JsonProcessor.load_json(ngram_stats_file)
    else:
        logger.info("Computing n-gram statistics...")
        collector = NGramContextCollector()
        try:
            collector.collect_stats(args.dataset, args.split)
        except ValueError as e:
            logger.error(f"Error processing dataset: {e}")
            return

        ngram_stats = collector.get_all_ngram_stats()
        JsonProcessor.save_json(ngram_stats, ngram_stats_file)
        logger.info(f"Saved n-gram statistics to {ngram_stats_file}")

    selected_contexts = NGramContextCollector.filter_contexts(
        ngram_stats, target_words, args.n_contexts, args.mode, args.window_size
    )

    JsonProcessor.save_json(selected_contexts, selected_contexts_file)
    logger.info(f"Saved selected contexts to {selected_contexts_file}")


if __name__ == "__main__":
    main()
