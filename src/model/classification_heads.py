#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

from typing import NamedTuple, Optional
import torch
from allennlp.common import Registrable
from allennlp.nn.util import sequence_cross_entropy_with_logits
from overrides import overrides
from torch import nn

from model.activations import Swish


class ClassificationOut(NamedTuple):
    logits: torch.tensor
    prediction_probabilities: torch.tensor
    loss: torch.tensor


class ClassificationHead(nn.Module, Registrable):

    def __init__(self, encoder_output_size: Optional[int], vocab_size: Optional[int], optimize_on_training: bool) -> None:
        super().__init__()
        self.encoder_output_size = encoder_output_size
        self.vocab_size = vocab_size
        self.optimize_on_training = optimize_on_training

    def forward(self, encoder_out: torch.Tensor, target: Optional[torch.tensor], target_mask: torch.tensor) -> ClassificationOut:
        
        batch_size, sentence_len, _ = encoder_out.shape
        classification_input = encoder_out
        
        if self.training and self.optimize_on_training:
            # optimize: compute head only for needed timesteps
            optimized_classification_input = encoder_out.view(-1, self.encoder_output_size)[target_mask.view(-1)]
            optimized_classification_logits = self.compute_logits(optimized_classification_input)
            classification_logits = encoder_out.new_zeros(batch_size, sentence_len, self.vocab_size)
            classification_logits.view(-1, self.vocab_size)[target_mask.view(-1)] = optimized_classification_logits
        else:
            classification_logits = self.compute_logits(classification_input)

        prediction_probabilities = torch.nn.functional.softmax(classification_logits, dim=-1)
        loss = None

        if target is not None:
            target[~target_mask] = -100
            loss = torch.nn.functional.cross_entropy(classification_logits.view(-1, self.vocab_size), target.view(-1))

        return ClassificationOut(classification_logits, prediction_probabilities, loss)

    def compute_logits(self, encoder_out: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@ClassificationHead.register('linear')
class LinearClassificationHead(ClassificationHead):

    def __init__(
            self,
            encoder_output_size: Optional[int],
            vocab_size: Optional[int],
            dropout: float = 0.5,
            output_bias: bool = False,
            optimize_on_training: bool = False
    ) -> None:
        super().__init__(encoder_output_size, vocab_size, optimize_on_training)
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_output_size, vocab_size, bias=output_bias)
        )

    @overrides
    def compute_logits(self, encoder_out: torch.Tensor) -> torch.Tensor:
        return self.classification_head(encoder_out)


@ClassificationHead.register('tanh')
class TanhClassificationHead(ClassificationHead):

    def __init__(
            self,
            encoder_output_size: Optional[int],
            vocab_size: Optional[int],
            dropout: float = 0.5,
            output_bias: bool = False,
            optimize_on_training: bool = False
    ) -> None:
        super().__init__(encoder_output_size, vocab_size, optimize_on_training)
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_output_size, encoder_output_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(encoder_output_size, vocab_size, bias=output_bias)
        )

    @overrides
    def compute_logits(self, encoder_out: torch.Tensor) -> torch.Tensor:
        return self.classification_head(encoder_out)


@ClassificationHead.register('swish')
class SwishClassificationHead(ClassificationHead):

    def __init__(
            self,
            encoder_output_size: Optional[int],
            vocab_size: Optional[int],
            dropout: float = 0.1,
            output_bias: bool = False,
            optimize_on_training: bool = False
    ) -> None:
        super().__init__(encoder_output_size, vocab_size, optimize_on_training)
        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.Linear(encoder_output_size, encoder_output_size)
        self.norm = nn.BatchNorm1d(encoder_output_size)
        self.swish = Swish()
        self.l2 = nn.Linear(encoder_output_size, vocab_size, bias=output_bias)

    @overrides
    def compute_logits(self, encoder_out: torch.Tensor) -> torch.Tensor:

        out = encoder_out

        # dropout + l1
        out = self.dropout(out)
        out = self.l1(out)

        # batch norm
        shape = out.shape
        out = self.norm(out.view(-1, shape[-1])).view(*shape)

        # swish
        out = self.swish(out)

        # dropout + l2
        out = self.dropout(out)
        out = self.l2(out)

        return out
