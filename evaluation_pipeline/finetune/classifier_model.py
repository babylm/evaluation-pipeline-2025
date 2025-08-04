from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import ModelOutput

if TYPE_CHECKING:
    from argparse import Namespace


class ClassifierHead(nn.Module):

    def __init__(self: ClassifierHead, config: Namespace, hidden_size: int | None = None) -> None:
        """This is the class for the classification head when doing
        sentence/sequence classification. This uses a config object
        to create the classification head for a certain task with a
        given pre-trained model.

        Args:
            config(Namespace): Contains all the information to create
                the classification head, including the number of
                classes for the task.
            hidden_size(int | None): The hidden size of the
                pre-trained model. If it is None, it is assumed that
                the config object contains the hidden size.
        """
        super().__init__()
        hidden_size: int = hidden_size if hidden_size is not None else config.hidden_size
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(hidden_size, config.classifier_layer_norm_eps, elementwise_affine=False),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, config.classifier_layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden_size, config.num_labels)
        )

    def forward(self: ClassifierHead, encodings: torch.Tensor) -> torch.Tensor:
        """This function handles the forward call of the
        classification head. It takes the model encodings and
        gives the logits for each class.

        Args:
            encodings(torch.Tensor): A tensor containing a the
            model encodings of the data used to classify.

        Returns:
            torch.Tensor: The logits for each class based on
                the encodings of the model for a given input.

        Shapes:
            - encodings: :math:`(B, S, D)`
        """
        return self.nonlinearity(encodings)


class ModelForSequenceClassification(nn.Module):

    def __init__(self: ModelForSequenceClassification, config: Namespace) -> None:
        """This is class create extends a pre-trained language model to
        classification tasks. This requires fine-tuning since the head
        is randomly generated. The model handles multiple output types
        of the pre-trained langauge model and whether to pass the first
        or last token to the classification head.

        Args:
            config(Namespace): Contains all the information to create
                the classification model, including the path to the
                pre-trained model and whether to pass the first or
                last token to the classification head.
        """
        super().__init__()
        self.transformer: nn.Module = AutoModel.from_pretrained(config.model_name_or_path, trust_remote_code=True, revision=config.revision_name)
        self.enc_dec: bool = config.enc_dec
        self.causal: bool = config.causal
        model_config = AutoConfig.from_pretrained(config.model_name_or_path, trust_remote_code=True, revision=config.revision_name)
        if self.enc_dec:
            self.decoder_start_token_id = model_config.decoder_start_token_id
        hidden_size = model_config.hidden_size
        self.classifier: nn.Module = ClassifierHead(config, hidden_size)
        self.take_final: bool = config.take_final

    def forward(self: ModelForSequenceClassification, input_data: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """This function handles the forward call of the model. It
        takes input data and mask and gives the logits for each class.

        Args:
            input_data(torch.Tensor): A tensor containing a batch
                of tokenized sentences (or pairs of sentences) to
                classify.
            attention_mask(torch.Tensor | None): A tensor of 1s and
                0s representing which tokens to attend to. If it is
                None, all tokens are attended to.

        Returns:
            torch.Tensor: The logits given by the model for each
                class based on the inputs.

        Shapes:
            - input_data: :math:`(B, S)`
            - attention_mask: :math:`(B, S)` or :math:`(B, S, S)`
        """
        if self.enc_dec:
            batch_size = attention_mask.size(0)
            decoder_input_ids = input_data.new_full((batch_size, 1), self.decoder_start_token_id)
            decoder_attention_mask = attention_mask.new_ones((batch_size, 1))
            output_transformer: Any = self.transformer(input_ids=input_data, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
        else:
            output_transformer = self.transformer(input_ids=input_data, attention_mask=attention_mask)
        if type(output_transformer) is tuple:
            encoding: torch.Tensor = output_transformer[0]
        elif isinstance(output_transformer, ModelOutput):
            if hasattr(output_transformer, "logits"):
                encoding = output_transformer.logits
            elif hasattr(output_transformer, "last_hidden_state"):
                encoding = output_transformer.last_hidden_state
            elif hasattr(output_transformer, "hidden_states"):
                encoding = output_transformer.hidden_states[-1]
            else:
                print("Unknown name for output of the model!")
                exit()
        else:
            print(f"Add support for output type: {type(output_transformer)}!")
            exit()
        if self.take_final and not self.enc_dec and not self.causal:
            transformer_output: torch.Tensor = encoding[:, -1]
        elif self.take_final and not self.enc_dec:
            final_position: torch.Tensor = attention_mask[:, :, -1].squeeze().long().argmax(-1) - 1
            transformer_output: torch.Tensor = encoding[final_position].diagonal().t()
        else:
            transformer_output = encoding[:, 0]
        logits: torch.Tensor = self.classifier(transformer_output)

        return logits
