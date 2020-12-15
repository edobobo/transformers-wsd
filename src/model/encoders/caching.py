#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

from typing import List, Tuple, Optional

import numpy as np
import torch
from allennlp.common import Lazy
from overrides import overrides

from model.encoders.base import Encoder
from model.training_strategies import TrainingStrategy


class CachedEncoder(Encoder):

    def __init__(self, backend_encoder: Encoder, cache_only_on_training: bool = True):
        super().__init__()
        self.backend_encoder = backend_encoder
        self._cache_only_on_training = cache_only_on_training

    def encode_tokens(self, tokens: List[str]) -> Tuple[np.array, List[Tuple[int, int]]]:
        return self.backend_encoder.encode_tokens(tokens)

    def get_output_size(self) -> int:
        return self.backend_encoder.get_output_size()

    @overrides
    def forward(self, sentences: torch.Tensor, padding_mask: torch.Tensor, tokens_offsets: List[List[Tuple[int, int]]], sentence_ids: List[str], sense_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if sense_mask is not None:
            if self.training or not self._cache_only_on_training:
                return self._forward(sentences, padding_mask, tokens_offsets, sentence_ids, sense_mask)
        return self.backend_encoder(sentences, padding_mask, tokens_offsets, sentence_ids, sense_mask)

    @overrides
    def _forward(self, sentences: torch.Tensor, padding_mask: torch.Tensor, tokens_offsets: List[List[Tuple[int, int]]], sentence_ids: List[str], sense_mask: torch.Tensor) -> torch.Tensor:
        return super().forward(sentences, padding_mask, tokens_offsets, sentence_ids, sense_mask)


@Encoder.register('sense-mask-only-cached-encoder')
class SenseMaskOnlyCachedEncoder(CachedEncoder):

    def __init__(self, backend_encoder: Encoder, cache_only_on_training: bool = True):
        super().__init__(backend_encoder, cache_only_on_training)
        self._cache = []
        self._sentence_id_cache_mapping = {}

    def _forward(self, sentences: torch.Tensor, padding_mask: torch.Tensor, tokens_offsets: List[List[Tuple[int, int]]], sentence_ids: List[str], sense_mask: torch.Tensor) -> torch.Tensor:

        batch_size, batch_token_len = sense_mask.shape
        encoder_size = self.backend_encoder.get_output_size()

        if any(sentence_id not in self._sentence_id_cache_mapping for sentence_id in sentence_ids):

            encoder_out_1 = self.backend_encoder(sentences, padding_mask, tokens_offsets, sentence_ids, sense_mask)

            for i, (sentence_id, _sense_mask) in enumerate(zip(sentence_ids, sense_mask)):

                if sentence_id not in self._sentence_id_cache_mapping:

                    self._sentence_id_cache_mapping[sentence_id] = {}

                    for timestep, _b in enumerate(_sense_mask):
                        if _b:
                            self._sentence_id_cache_mapping[sentence_id][timestep] = len(self._cache)
                            self._cache.append(encoder_out_1[i, timestep].detach().cpu())

        encoder_out = sentences.new_zeros((batch_size, batch_token_len, encoder_size), dtype=torch.float)

        for i, sentence_id in enumerate(sentence_ids):
            for timestep, cache_row in self._sentence_id_cache_mapping[sentence_id].items():
                encoder_out[i, timestep] = self._cache[cache_row].to(encoder_out.device)

        return encoder_out


@Encoder.register('general-cached-encoder')
class SenseMaskOnlyCachedEncoder(CachedEncoder):

    def _forward(self, sentences: torch.Tensor, padding_mask: torch.Tensor, tokens_offsets: List[List[Tuple[int, int]]], sentence_ids: List[str], sense_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
