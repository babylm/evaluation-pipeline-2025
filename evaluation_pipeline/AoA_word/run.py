#!/usr/bin/env python
import argparse
import logging
import pathlib
from pathlib import Path

import torch

from evaluation_pipeline.AoA_word.eval_util import JsonProcessor, StepConfig, load_eval
from evaluation_pipeline.AoA_word.evaluation_functions import StepSurprisalExtractor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract word surprisal across different training steps."
    )
    parser.add_argument(
        "-w",
        "--word_path",
        type=Path,
        default="evaluation_data/full_eval/cdi_childes/cdi_childes.json",
        help="Relative path to the target words",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="gpt2",
        help="Model directory with different steps",
    )
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default="causal",
        choices=["mlm", "causal", "mntp", "enc_dec_mask", "enc_dec_prefix"],
    )
    parser.add_argument(
        "-t",
        "--track_name",
        type=str,
        default="non-strict-small",
        choices=["strict-small", "non-strict-small"],
        help="Which track the model was trained for (controls checkpoint names)",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        type=pathlib.Path,
        help="Path to the data directory",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Compute the first 5 lines if enabled"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume results from the existing checkpoint",
    )
    parser.add_argument(
        "--min_context",
        default=0,
        type=int,
        help="Minimum number of contexts for a given word to be evaluated"
    )
    return parser.parse_args()


def config_paths(args) -> tuple[Path, Path | None]:
    """Initialize paths for results and resume files."""
    model_name = pathlib.Path(args.model_name).stem
    full_output_dir = (
        args.output_dir / model_name / "main" / "zero_shot" / args.backend / "AoA_word"
    )
    full_output_dir.mkdir(parents=True, exist_ok=True)
    print("Saving AoA results to: ", full_output_dir)

    result_file = full_output_dir / "surprisal.json"
    if args.resume:
        resume_file = full_output_dir / "resume" / "surprisal.json"
        resume_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        resume_file = None

    return result_file, resume_file


def save_results(results_data: dict, result_file: Path) -> None:
    """Save results to JSON file."""
    if results_data and "results" in results_data and results_data["results"]:
        JsonProcessor.save_json(results_data, result_file)

        completed_steps = len({r["step"] for r in results_data["results"]})
        logger.info(
            f"Results saved to: {result_file}\n"
            f"Processed {completed_steps} checkpoints successfully"
        )
    else:
        logger.warning("No results were generated")


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target_words, contexts = load_eval(args.word_path, args.min_context, args.debug)
    result_file, resume_file = config_paths(args)

    steps_config = StepConfig(
        resume=args.resume,
        track=args.track_name,
        file_path=resume_file,
        debug=args.debug,
    )

    extractor = StepSurprisalExtractor(
        config=steps_config,
        model_name=args.model_name,
        backend=args.backend,
        device=device,
    )

    logger.info("Computing surprisal across training steps")
    results_data = extractor.analyze_steps(
        contexts=contexts,
        target_words=target_words,
        resume_path=resume_file,
    )

    save_results(results_data, result_file)


if __name__ == "__main__":
    main()
