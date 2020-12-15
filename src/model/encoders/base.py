#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import random
from typing import Tuple, List, Optional

import numpy as np
import torch
from allennlp.common import Registrable, Lazy
from torch import nn

from model.training_strategies import TrainingStrategy


class Encoder(nn.Module, Registrable):

    def encode_tokens(self, tokens: List[str]) -> Tuple[np.array, List[Tuple[int, int]]]:
        raise NotImplementedError

    def get_output_size(self) -> int:
        raise NotImplementedError

    def forward(self, sentences: torch.Tensor, padding_mask: torch.Tensor, tokens_offsets: List[List[Tuple[int, int]]], sentence_ids: List[str], sense_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.training:
            self.training_strategy.step()
        return self._forward(sentences, padding_mask, tokens_offsets, sentence_ids, sense_mask)

    def _forward(self, sentences: torch.Tensor, padding_mask: torch.Tensor, tokens_offsets: List[List[Tuple[int, int]]], sentence_ids: List[str], sense_mask: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
