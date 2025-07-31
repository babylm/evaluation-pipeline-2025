from __future__ import annotations

import torch
import argparse
import json
import pathlib

from evaluation_pipeline.finetune.trainer import Trainer
from evaluation_pipeline.finetune.utils import seed_everything

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("The argument parser for the finetuning pipeline, it takes all hyperparameters, model information, data information, and save directories.")

    # Required Parameters
    parser.add_argument("--results_dir", default="results", type=pathlib.Path, help="The output directory where the results will be written.")
    parser.add_argument("--train_data", required=True, type=pathlib.Path, help="Path to file containing the training dataset, we expect it to be in a JSONL format.")
    parser.add_argument("--model_name_or_path", required=True, type=str, help="The local path to the model binary.")
    parser.add_argument("--metrics", default=["accuracy"], nargs='+', help="List of metrics to evaluate for the model (accuracy, f1, and mcc).", choices=["accuracy", "f1", "mcc"])
    parser.add_argument("--num_labels", required=True, type=int, help="The number of labels in the dataset. (3 for MNLI, 2 for all other tasks)")
    parser.add_argument("--seed", default=42, type=int, help="The seed for the Random Number Generator.")
    parser.add_argument("--task", required=True, type=str, help="The task to fine-tune for.")

    # Optinal Parameters
    parser.add_argument("--ema_decay", default=0.0, type=float, help="If using EMA, this is the decay rate per step. (If it is 0 then there is no ema_decay)")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Whether to output the metrics in terminal during the run.")
    parser.add_argument("--valid_data", type=pathlib.Path, help="Path to file containing the validation dataset to validate on, we expect it to be in a JSONL format.")
    parser.add_argument("--predict_data", type=pathlib.Path, help="Path to file containing the dataset to predict on, we expect it to be in a JSONL format.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the fine-tuned model.")
    parser.add_argument("--save_dir", default="finetuned-models", type=pathlib.Path, help="The directory in which to save the fine-tuned model.")
    parser.add_argument("--keep_best_model", action=argparse.BooleanOptionalAction, default=True, help="Whether to keep the model with the best score based on the metric_for_valid. (If False, then the model at the end of fine-tuning will be used for eval and prediction)")
    parser.add_argument("--metric_for_valid", type=str, help="The metric used to compare the model when finding the best model.", choices=["accuracy", "mcc", "f1"])
    parser.add_argument("--higher_is_better", action=argparse.BooleanOptionalAction, default=True, help="Wheter a higher value for the metric for valid is better or not.")
    parser.add_argument("--revision_name", default=None, type=str, help="Name of the checkpoint/version of the model to test. (If None, the main will be used)")

    # Hyperparameters
    parser.add_argument("--batch_size", default=16, type=int, help="The batch size for each step during fine-tuning.")
    parser.add_argument("--gradient_accumulation", default=1, type=int, help="The number of graident accumualtion steps to do, to reach the desired batch size (needs to divide the batch size argument).")
    parser.add_argument("--valid_batch_size", default=64, type=int, help="The batch size during inference.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The maximum learning rate during fine-tuning.")
    parser.add_argument("--sequence_length", default=512, type=int, help="The max sequence length before truncation.")
    parser.add_argument("--num_epochs", default=10, type=int, help="Number of epochs to fine-tune the code for.")
    parser.add_argument("--classifier_dropout", default=0.1, type=float, help="The dropout applied to the classifier head. (Needs to be a value between 0 and 1)")
    parser.add_argument("--classifier_layer_norm_eps", default=1.0e-5, type=float, help="The epsilon to add to the layer norm operations to stabalize the division and avoid dividing by zero.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="The weight decay to apply for the optimizer (if a weight decay is relevant). (Needs to be a value between 0 and 1)")
    parser.add_argument("--warmup_proportion", default=0.06, type=float, help="The proportion of the fine-tuning steps where the learning rate increases from 0 to its maximum value. (Needs to be a value between 0 and 1)")
    parser.add_argument("--min_factor", default=0.1, type=float, help="The final factor to which the max learning rate is multiplied to find the final learning rate.")
    parser.add_argument("--scheduler", default="cosine", type=str, help="The learning rate scheduler to use for fine-tuning. none means that no learning rate scheduling was chosen.", choices=["cosine", "none"])  # Not implemented
    parser.add_argument("--optimizer", default="adamw", type=str, help="The optimizer to use for the fine-tuning of the model.", choices=["adamw", "adam"])  # Not implemented
    parser.add_argument("--beta1", default=0.9, type=float, help="The value of beta1 (or beta) in optimizers that require it.")
    parser.add_argument("--beta2", default=0.999, type=float, help="The value of beta2 in optimizers that require it.")
    parser.add_argument("--optimizer_eps", default=1e-8, type=float, help="The epsilon to add to the optimizer operations (if relevant) to stabalize and avoid dividing by zero.")
    parser.add_argument("--padding_side", default='right', type=str, help="The padding side to use for the tokenizer")
    parser.add_argument("--amsgrad", default=False, action=argparse.BooleanOptionalAction, help="Whether to use AMSGrad variant of the AdamW optimizer. (Only relevant if adamw chosen for optimizer)")
    parser.add_argument("--causal", default=False, action=argparse.BooleanOptionalAction, help="Whether to use causal masking")
    parser.add_argument("--take_final", default=False, action=argparse.BooleanOptionalAction, help="Whether to take the last token rather than the first one.")
    parser.add_argument("--enc_dec", default=False, action=argparse.BooleanOptionalAction, help="Flag to indicate that the architecture is an encoder-decoder.")

    # W&B parameters
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, help="Flag to activate W&B logging.")
    parser.add_argument("--wandb_project", type=str, default="BabyLM Finetuning", help="The name of the project to log to. By default this is BabyLM Finetuning.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="The name of the user/organization on W&B to log to.")
    parser.add_argument("--exp_name", type=str, default=None, help="The name of the run as it appears on W&B. By default this is: 'model_name_task_seed'")

    args = parser.parse_args()
    args.model_name = pathlib.Path(args.model_name_or_path).stem

    if args.wandb and args.exp_name is None:
        args.exp_name = "_".join([args.model_name, args.task, str(args.seed)])

    return args


if __name__ == "__main__":
    args: Namespace = _parse_arguments()

    seed_everything(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_name: str = args.model_name
    if args.task == "mnli":
        if args.valid_data is not None:
            args.task: str = args.valid_data.stem.split(".")[0]
        elif args.predict_data is not None:
            args.task: str = args.predict_data.stem.split(".")[0]
    if args.revision_name is None:
        revision_name = "main"
    else:
        revision_name = args.revision_name
    output_path: pathlib.Path = args.results_dir / model_name / revision_name / "finetune" / args.task
    output_path.mkdir(parents=True, exist_ok=True)
    if args.save:
        args.save_path: pathlib.Path = args.save_dir / model_name / args.task
        args.save_path.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(args, device)
    trainer.train()

    if trainer.valid_dataloader is not None:
        metrics: dict[str, float] = trainer.evaluate(evaluate_best_model=True)
        with (output_path / "results.txt").open("w") as file:
            file.write("\n".join([f"{key}: {value}" for key, value in metrics.items()]))

    if trainer.predict_dataloader is not None:
        preds: torch.Tensor = trainer.predict_classification()
        pred_dict: dict[str, dict[str, list[dict[str, str | float]]]] = {f"{args.task}": {"predictions": []}}
        for i, pred in enumerate(preds):
            pred_dict[f"{args.task}"]["predictions"].append({"id": f"{args.task}_{i}", "pred": int(pred)})
        with (output_path / "predictions.json").open("w") as file:
            json.dump(pred_dict, file)

    if args.wandb:
        trainer.wandb_run.finish()
