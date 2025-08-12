# File: run_zero_shot.py
# ----------------------
from __future__ import annotations

import pathlib
import json
import argparse
from _io import TextIOWrapper

from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
import torch

from evaluation_pipeline.sentence_zero_shot.dataset import get_dataloader
from evaluation_pipeline.sentence_zero_shot.compute_results import compute_results

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def _parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_path", required=True, type=pathlib.Path, help="Path to the data directory")
    parser.add_argument("--task", required=True, type=str, help="The task that is being evaluated.", choices=["blimp", "ewok", "entity_tracking", "wug_adj", "wug_past", "comps", "vqa", "winoground"])
    parser.add_argument("--model_path_or_name", required=True, type=str, help="Path to the model to evaluate.")
    parser.add_argument("--backend", required=True, type=str, help="The evaluation backend strategy", choices=["mlm", "causal", "mntp", "enc_dec_mask", "enc_dec_prefix"])

    parser.add_argument("--output_dir", default="results", type=pathlib.Path, help="Path to the data directory")
    parser.add_argument("--images_path", default=None, type=str, help="Path or HuggingFace repository name to the images for the task.")
    parser.add_argument("--image_split", default=None, type=str, help="The data split from a HuggingFace repository to use for the images source.")
    parser.add_argument("--image_template", default=None, type=str, help="Template of how to imbed the image to the text. (In cases where this is not handled within the model).")

    parser.add_argument("--revision_name", default=None, type=str, help="Name of the checkpoint/version of the model to test. (If None, the main will be used)")

    parser.add_argument("--min_temperature", default=1.0, type=float, help="Minimum temperature to apply to the logits.")
    parser.add_argument("--max_temperature", default=None, type=float, help="Maximum temperature to apply to the logits. If None, onlny the minimum temperature will be considered.")
    parser.add_argument("--temperature_interval", default=0.05, type=float, help="Step size between temperatures applied to the logits.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for evaluation")
    parser.add_argument("--non_causal_batch_size", default=64, type=int, help="Mini-batch size to process each batch of inputs involving masked tokens")
    parser.add_argument("--full_sentence_scores", action="store_true", help="Whether to use the entire sentence to calculate the sentence scores rather than just the completion. (Only implemented for EWoK)")
    parser.add_argument("--save_predictions", action="store_true", help="Whether or not to save predictions.")

    return parser.parse_args()


def get_model(args: argparse.ArgumentParser):
    if args.backend in ["mlm", "mntp"]:
        model = AutoModelForMaskedLM.from_pretrained(args.model_path_or_name, trust_remote_code=True, revision=args.revision_name)
    elif args.backend == "causal":
        model = AutoModelForCausalLM.from_pretrained(args.model_path_or_name, trust_remote_code=True, revision=args.revision_name)
    elif args.backend in ["enc_dec_mask", "enc_dec_prefix"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path_or_name, trust_remote_code=True, revision=args.revision_name)
    else:
        raise f"The backend {args.backend} is not implemented, please implemented yourself or raise an issue on the GitHub!"
    model = model.to(DEVICE)
    model.eval()

    return model


def get_temperatures(args: argparse.ArgumentParser):
    if args.max_temperature is None:
        temperatures = torch.ones(1) * args.min_temperature
    else:
        temperatures = torch.arange(
            args.min_temperature,
            args.max_temperature + args.temperature_interval,
            args.temperature_interval,
        ).clamp(min=1e-6)
    return temperatures.tolist()


def process_results(args: argparse.ArgumentParser, results: dict):
    """This function computes accuracy metrics and, if necessary, other dataset-specific metrics
    given dataset sizes and numbers of correct predictions

    Args:
        args (argparse.ArgumentParser): ArgumentParser object used to determine task
        results (dict): Results obtained from running compute_results
    """
    # Compute accuracies
    accuracies = {temp : {} for temp in results}
    for temp, temp_results in results.items():
        for subdomain, count_dict in temp_results.items():
            keys = count_dict["total"].keys()
            subdomain_accs = {key : 100.0 * count_dict["correct"][key] / count_dict["total"][key] for key in keys}
            accuracies[temp][subdomain] = subdomain_accs

    # Average accuracies
    average_accuracies = {}
    if args.task != "entity_tracking":
        for temp, accuracy in accuracies.items():
            average_accuracies[temp] = sum(accuracy["UID"].values()) / len(accuracy["UID"].values())
    else:
        splits = ["regular", "ambiref", "move_contents"]
        for temp, subdomain_dict in accuracies.items():
            split_accs = []
            split_dict = subdomain_dict["UID"]
            for split in splits:
                split_keys = [key for key in split_dict if key.startswith(split)]
                if not split_keys:
                    continue
                curr_acc = sum([split_dict[key] for key in split_keys]) / len(split_keys)

                split_dict[split] = curr_acc
                split_accs.append(curr_acc)
            average_accuracies[temp] = sum(split_accs) / len(split_accs)

    return accuracies, average_accuracies


def process_results_wug(results):
    correlations = {temp : {} for temp in results}
    avg_correlations = {}

    for temp, temp_results in results.items():
        correlations[temp]["UID"] = {"avg": temp_results["correlation"]}
        avg_correlations[temp] = temp_results["correlation"]
        # for subdomain, subdomain_correlations in temp_results.items():
        #     correlations[temp][subdomain] = {"avg": subdomain_correlations["correlation"]}
        #     avg_correlations[temp] = subdomain_correlations["correlation"]

    return correlations, avg_correlations


def create_evaluation_report(temperature: float, avg_accuracy: torch.Tensor, accuracies: dict[str, list[dict[str, float]]], task: str | None = None, file: TextIOWrapper | None = None) -> None:
    """This function creates a report and either saves it to a file or prints it to the terminal.

    Args:
        temperature(float): The temperature at which the model is evaluated.
        temperature_pos(int): The position of the evaluated temperature.
        avg_accuracy(torch.Tensor): The average accuracy of the model at the given temperature.
        avg_accuracy(dict[str, list[dict[str, float]]]): The finegrained accuracies of the model
            at the given temperature.
        file(TextIOWrapper | None): The file to write to results to. (If None, it will printed
            printed to the terminal)
    """
    metric = "ACCURACY" if "wug" not in task else "SPEARMAN'S RHO"
    print(f"TEMPERATURE: {temperature:.2f}", file=file)
    print(file=file)

    for domain, accuracy in accuracies.items():
        print(f"### {domain.upper()} {metric}", file=file)
        for subdomain, acc in accuracy.items():
            print(f"{subdomain}: {acc:.2f}", file=file)
        print(file=file)

    print(f"### AVERAGE {metric}", file=file)
    print(f"{avg_accuracy:.2f}", file=file)
    print(file=file)


def save_predictions(args, predictions, best_temp):
    with (args.output_path / "predictions.json").open("w") as f:
        json.dump(predictions[best_temp], f)


def main():
    args = _parse_arguments()
    if args.images_path is not None:
        assert args.batch_size == 1, "Multimodal only works in batch size 1!"
    dataset = args.data_path.stem
    args.model_name = pathlib.Path(args.model_path_or_name).stem
    if args.revision_name is None:
        revision_name = "main"
    else:
        revision_name = args.revision_name
    args.output_path = args.output_dir / args.model_name / revision_name / "zero_shot" / args.backend / args.task / dataset
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Get results
    model = get_model(args)
    dataloader = get_dataloader(args)
    temperatures = get_temperatures(args)
    results, predictions = compute_results(args, model, dataloader, temperatures)

    # Process results
    if "wug" in args.task:
        accuracies, average_accuracies = process_results_wug(results)
    else:
        accuracies, average_accuracies = process_results(args, results)
    best_acc = -1
    best_temp = -1
    for temperature, acc in average_accuracies.items():
        print(f"{temperature}\t{acc:.2f}")
        if acc > best_acc:
            best_acc = acc
            best_temp = temperature
    print()

    # Report and save
    create_evaluation_report(best_temp, average_accuracies[best_temp], accuracies[best_temp], task=args.task)
    with (args.output_path / "best_temperature_report.txt").open("w") as f:
        create_evaluation_report(best_temp, average_accuracies[best_temp], accuracies[best_temp], task=args.task, file=f)

    # Save predictions
    if args.save_predictions:
        save_predictions(args, predictions, best_temp)


if __name__ == "__main__":
    main()
