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
        default="context/stas/c4-en-10k/5/merged.json",
        help="Relative path to the target words",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="EleutherAI/pythia-70m-deduped",
        help="Target model name",
    )
    parser.add_argument(
        '-b',
        '--backend',
        type=str,
        default="causal",
        choices=["mlm", "causal", "mntp", "enc_dec_mask", "enc_dec_prefix"]
    )
    parser.add_argument(
        "-t",
        "--track_name",
        type=str,
        default="non-strict-small", choices=["strict-small", "non-strict-small"],
        help="Which track the model was trained for (controls checkpoint names)"
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        type=pathlib.Path,
        help="Path to the data directory"
    )
    parser.add_argument(
        "--eval_lst",
        type=list,
        help="Eval file list",
    )
    parser.add_argument(
        "--min_context", type=int, default=20, help="Minimum number of contexts"
    )
    parser.add_argument(
        "--use_bos_only", action="store_true", help="Use BOS only if enabled"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Compute the first 5 lines if enabled"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the existing checkpoint"
    )
    return parser.parse_args()


def config_paths(args) -> tuple[Path, Path | None]:
    """Initialize paths for results and resume files."""
    model_name = pathlib.Path(args.model_name).stem
    full_output_dir = args.output_dir / model_name / "main" / "zero_shot" / args.backend / "AoA_word"
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


def create_eval_subset(
    results_data: dict, eval_paths: list[Path], output_dir: Path
) -> None:
    """Create evaluation subsets from results data."""
    if not eval_paths or not results_data.get("results"):
        return

    for eval_path in eval_paths:
        if not eval_path.exists():
            logger.warning(f"Eval file not found: {eval_path}")
            continue

        try:
            eval_data = JsonProcessor.load_json(eval_path)
            if isinstance(eval_data, dict) and "words" in eval_data:
                eval_words = set(eval_data["words"])
            elif isinstance(eval_data, list):
                eval_words = set(eval_data)
            else:
                logger.warning(f"Unexpected eval file format: {eval_path}")
                continue

            filtered_results = [
                result
                for result in results_data["results"]
                if result["target_word"] in eval_words
            ]

            if filtered_results:
                eval_output = {
                    "metadata": results_data["metadata"],
                    "eval_source": str(eval_path),
                    "results": filtered_results,
                }

                eval_filename = f"{eval_path.stem}_subset.json"
                eval_output_path = output_dir / eval_filename
                JsonProcessor.save_json(eval_output, eval_output_path)
                logger.info(f"Eval subset saved to: {eval_output_path}")

        except Exception as e:
            logger.error(f"Error processing eval file {eval_path}: {e}")


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
        use_bos_only=args.use_bos_only,
        resume_path=resume_file,
    )

    save_results(results_data, result_file)

    if args.eval_lst:
        create_eval_subset(results_data, args.eval_lst, result_file.parent)


if __name__ == "__main__":
    main()
