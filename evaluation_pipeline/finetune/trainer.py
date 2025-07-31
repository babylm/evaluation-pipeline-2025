from __future__ import annotations

from tqdm import tqdm
from typing import TYPE_CHECKING
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import copy
import pathlib
from functools import partial

import torch
from torch.nn import functional as F
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from torch.optim import AdamW

from evaluation_pipeline.finetune.classifier_model import ModelForSequenceClassification
from evaluation_pipeline.finetune.dataset import Dataset, PredictDataset
from evaluation_pipeline.finetune.utils import cosine_schedule_with_warmup

if TYPE_CHECKING:
    from argparse import Namespace
    import torch.nn as nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import wandb


def _load_labeled_dataset(data_path: pathlib.Path, batch_size: int, tokenizer: PreTrainedTokenizerBase, shuffle: bool, drop_last: bool, args: Namespace) -> DataLoader:
    dataset = Dataset(data_path, args.task)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(dataset.collate_function, tokenizer, args.causal, args.sequence_length), shuffle=shuffle, drop_last=drop_last)

    return dataloader


def _load_predict_dataset(data_path: pathlib.Path, batch_size: int, tokenizer: PreTrainedTokenizerBase, args: Namespace):
    dataset = PredictDataset(data_path, args.task)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(dataset.collate_function, tokenizer, args.causal, args.sequence_length))

    return dataloader


class Trainer():

    def __init__(self: Trainer, args: Namespace, device: torch.device) -> None:
        """The Trainer class handles all the fine tuning,
        evaluation, and prediction of a given task for a
        given model.

        Args:
            args(Namespace): The config information such
                as hyperparameters, directories, verbose,
                model_name, optimizer_name, etc.
            device(torch.device): The device to use for
                finetuning.
        """
        self.args: Namespace = args
        self.device: torch.device = device
        self._init_model()
        self.load_data()
        self.global_step: int = 0
        self.total_steps = (len(self.train_dataloader) // self.args.gradient_accumulation) * self.args.num_epochs
        self._init_opitmizer()
        self._init_scheduler()
        if args.wandb:
            self._init_wandb()

    def _init_wandb(self: Trainer) -> None:
        self.wandb_run = wandb.init(
            name=self.args.exp_name,
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            config=self.args
        )

    def _init_model(self: Trainer) -> None:
        self.model = ModelForSequenceClassification(self.args)
        self.ema_model: nn.Module = copy.deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False

        self.model.to(self.device)
        self.ema_model.to(self.device)
        self.tokenizer: PreTrainedTokenizerBase = AutoProcessor.from_pretrained(self.args.model_name_or_path, trust_remote_code=True, padding_side=self.args.padding_side)

    def load_data(self: Trainer) -> None:
        """This function loads the data and creates the
        dataloader for each split of the data.
        """
        assert self.args.batch_size % self.args.gradient_accumulation == 0, f"The gradient accumualtion {self.args.gradient_accumulation} should divide the batch size {self.args.batch_size}."

        self.train_dataloader: DataLoader = _load_labeled_dataset(self.args.train_data, self.args.batch_size // self.args.gradient_accumulation, self.tokenizer, True, True, self.args)

        self.valid_dataloader: DataLoader | None = None
        if self.args.valid_data is not None:
            self.valid_dataloader: DataLoader = _load_labeled_dataset(self.args.valid_data, self.args.valid_batch_size, self.tokenizer, False, False, self.args)

        self.predict_dataloader: DataLoader | None = None
        if self.args.predict_data is not None:
            self.predict_dataloader: DataLoader = _load_predict_dataset(self.args.predict_data, self.args.valid_batch_size, self.tokenizer, self.args)

    def _init_opitmizer(self: Trainer) -> None:
        if self.args.optimizer in ["adamw", "adam"]:
            self.optimizer: Optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, betas=(self.args.beta1, self.args.beta2), eps=self.args.optimizer_eps, weight_decay=self.args.weight_decay, amsgrad=self.args.amsgrad)
        else:
            raise NotImplementedError(f"The optimizer {self.args.optimizer} is not implemented!")

    def _init_scheduler(self: Trainer) -> None:
        if self.args.scheduler == "cosine":
            self.scheduler: LRScheduler | None = cosine_schedule_with_warmup(self.optimizer, int(self.args.warmup_proportion * self.total_steps), self.total_steps, 0.1)
        elif self.args.scheduler == "none":
            self.scheduler = None
        else:
            raise NotImplementedError(f"The scheduler {self.args.scheduler} is not implemented!")

    # TODO: Create getter and setter functions

    def reset_trainer(self: Trainer) -> None:
        """This function resets the Trainer. This means that it
        resets the global step back to zero, and re-initializes
        the optimizer and scheduler."""
        self.global_step = 0
        self._init_opitmizer()
        self._init_scheduler()

    def train_epoch(self: Trainer) -> None:
        """This function does a single epoch of the training.

        Args:
            total_steps(int): The total number of finetuning
                steps to do. Used for the progress bar and in
                case the finetuning needs to be stoped mid
                epoch.
            global_step(int): The current step the finetuning
                is on.

        Returns:
            int: The current step the model is on at the end of
                the epoch.
        """
        self.model.train()
        self.optimizer.zero_grad()

        progress_bar = tqdm(initial=self.global_step, total=self.total_steps)
        cummulator = 0

        for input_data, attention_mask, labels in self.train_dataloader:
            input_data = input_data.to(device=self.device)
            attention_mask = attention_mask.to(device=self.device)
            labels = labels.to(device=self.device)

            logits = self.model(input_data, attention_mask)

            loss = F.cross_entropy(logits, labels)
            loss.backward()
            cummulator += 1
            if cummulator < self.args.gradient_accumulation:
                continue
            cummulator = 0

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            if self.ema_model is not None:
                with torch.no_grad():
                    for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
                        param_k.data.mul_(self.args.ema_decay).add_((1.0 - self.args.ema_decay) * param_q.detach().data)

            metrics = self.calculate_metrics(logits, labels, self.args.metrics)

            if hasattr(self, "wandb_run"):
                self.wandb_run.log(
                    {f"train/{metric}": value for metric, value in metrics.items()},
                    step=self.global_step
                )

            metrics_string = ", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])

            progress_bar.update()

            if self.args.verbose:
                progress_bar.set_postfix_str(metrics_string)

            self.global_step += 1
            self.optimizer.zero_grad()

        progress_bar.close()

    @torch.no_grad()
    def evaluate(self: Trainer, evaluate_best_model: bool = False) -> dict[str, float]:
        """This function does an evaluation pass on the
        validation dataset.

        Returns:
            dict[str, float]: A dictionary of scores of the
                model on the validation dataset, based on
                the metrics to evaluate on.
        """
        assert self.valid_dataloader is not None, "No valid dataset to run evaluation on!"

        if hasattr(self, "best_model") and evaluate_best_model:
            model: nn.Module = self.best_model
        elif self.ema_model is not None:
            model = self.ema_model
        else:
            model = self.model
        model.eval()

        progress_bar = tqdm(total=len(self.valid_dataloader))

        labels = []
        logits = []

        for input_data, attention_mask, label in self.valid_dataloader:
            input_data = input_data.to(device=self.device)
            attention_mask = attention_mask.to(device=self.device)
            label = label.to(device=self.device)

            logit = model(input_data, attention_mask)

            logits.append(logit)
            labels.append(label)

            progress_bar.update()

        labels = torch.cat(labels, dim=0)
        logits = torch.cat(logits, dim=0)

        metrics = self.calculate_metrics(logits, labels, self.args.metrics)

        if hasattr(self, "wandb_run"):
            self.wandb_run.log(
                {f"evaluate/{metric}": value for metric, value in metrics.items()},
                step=self.global_step
            )

        progress_bar.close()

        if self.args.verbose:
            metrics_string = "\n".join([f"{key}: {value}" for key, value in metrics.items()])
            print(metrics_string)

        return metrics

    def save_model(self: Trainer, model: nn.Module) -> None:
        """This function saves the passed model to a file. The
        directory is specified inside the arguments passed to
        the constructor of the class.

        Args:
            model(nn.Module): The model to save.
        """
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), self.args.save_path / "model.pt")

    def _compare_scores(self: Trainer, best: float, current: float, bigger_better: bool) -> bool:
        if best is None:
            return True
        else:
            if current > best and bigger_better:
                return True
            elif current < best and not bigger_better:
                return True
            return False

    @staticmethod
    def calculate_metrics(logits: torch.Tensor, labels: torch.Tensor, metrics_to_calculate: list[str]) -> dict[str, float]:
        """This function calculates the metrics specified by
        the user. This is a static method and can be used
        without initializing a Trainer.

        Args:
            logits(torch.Tensor): A tensor of logits per class
                calculated by a model.
            labels(torch.Tensor): A tensor of correct labels
                for each element of the batch
            metrics_to_calculate(list[str]): A list of metrics
                to evaluate.

        Returns:
        dict[str, float]: a dictionary containing the scores of
            the model on the specified metrics.

        Shapes:
            - logits: :math:`(B, N)`
            - labels: :math:`(B)`, where each element is in
                :math:`[0, N-1]`
        """
        predictions = logits.argmax(dim=-1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        metrics = dict()

        for metric in metrics_to_calculate:
            if metric == "f1":
                metrics["f1"] = f1_score(labels, predictions)
            elif metric == "accuracy":
                metrics["accuracy"] = accuracy_score(labels, predictions)
            elif metric == "mcc":
                metrics["mcc"] = matthews_corrcoef(labels, predictions)
            else:
                print(f"Metric {metric} is unknown / not implemented. It will be skipped!")

        return metrics

    def train(self: Trainer) -> None:
        """This function does the training based on the
        hyperparameters, model, optimizer, scheduler specified
        to the constructor of the class.
        """
        best_score: float | None = None
        self.best_model: nn.Module | None = None
        update_best: bool = False

        for epoch in range(self.args.num_epochs):
            self.train_epoch()

            if self.valid_dataloader is not None:
                metrics: dict[str, float] = self.evaluate()
                if self.args.keep_best_model:
                    score: float = metrics[self.args.metric_for_valid]
                    if self._compare_scores(best_score, score, self.args.higher_is_better):
                        if self.ema_model is not None:
                            self.best_model = copy.deepcopy(self.ema_model)
                        else:
                            self.best_model = copy.deepcopy(self.model)
                        best_score = score
                        update_best = True

            if self.args.save:
                if self.args.keep_best_model and update_best:
                    self.save_model(self.best_model)
                    update_best = False
                elif self.ema_model is not None:
                    self.save_model(self.ema_model)
                elif not self.args.keep_best_model:
                    self.save_model(self.model)

    @torch.no_grad()
    def predict_classification(self: Trainer) -> torch.Tensor:
        """This function creates predictions for the prediction
        dataset.

        Returns:
            dict[str, float]: A dictionary of scores of the
                model on the validation dataset, based on
                the metrics to evaluate on.
        """
        assert self.predict_dataloader is not None, "No predict dataset to predict on!"

        if hasattr(self, "best_model"):
            model: nn.Module = self.best_model
        elif self.ema_model is not None:
            model = self.ema_model
        else:
            model = self.model
        model.eval()

        progress_bar = tqdm(total=len(self.predict_dataloader))

        logits = []

        for input_data, attention_mask in self.predict_dataloader:
            input_data = input_data.to(device=self.device)
            attention_mask = attention_mask.to(device=self.device)

            logit = model(input_data, attention_mask)

            logits.append(logit)

            progress_bar.update()

        logits = torch.cat(logits, dim=0)
        preds = logits.argmax(dim=-1)

        progress_bar.close()

        return preds
