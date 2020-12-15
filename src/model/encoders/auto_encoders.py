#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
from allennlp.common import Lazy
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel, AutoConfig

from model.encoders.base import Encoder
from model.training_strategies import TrainingStrategy


logger = logging.getLogger(__name__)


@Encoder.register('transformers')
class AutoEncoder(Encoder):

    _shared_state = {}

    def __init__(
            self,
            transformer_model: str,
            use_last_n_layers: int = 1,
            cls_start: bool = True,
            use_fast_tokenizer: bool = False,
            max_bpe: int = 512,
            training_strategy: Optional[Lazy[TrainingStrategy]] = None
    ) -> None:

        super().__init__()

        # key-based borg pattern

        if transformer_model not in self._shared_state:

            self._shared_state[transformer_model] = self.__dict__

            # load tokenizer
            self.auto_tokenizer = AutoTokenizer.from_pretrained(transformer_model, use_fast=use_fast_tokenizer)

            # load model
            auto_config = AutoConfig.from_pretrained(transformer_model)
            auto_config.output_hidden_states = True
            self.auto_model = AutoModel.from_pretrained(transformer_model, config=auto_config)

            # params
            self.use_last_n_layers = use_last_n_layers
            self.cls_start = cls_start
            self.max_bpe = max_bpe

            # training strategy
            self.training_strategy = training_strategy.construct(module=self)

        else:

            self.__dict__ = self._shared_state[transformer_model]

    # todo: this method is valid only for transformers that have a pre split at token level.
    def encode_tokens(self, tokens: List[str]) -> Tuple[np.array, List[Tuple[int, int]]]:

        sentence_bpes = []
        tokens_offsets = []

        for token in tokens:

            token_bpes = []

            for _token in token.split('_'):
                token_bpes += self.auto_tokenizer.tokenize(_token)

            curr_bpes_len = len(sentence_bpes) + (1 if self.cls_start else 0)
            tokens_offsets.append(
                (curr_bpes_len, curr_bpes_len + len(token_bpes))
            )

            sentence_bpes += token_bpes

        if len(sentence_bpes) > self.max_bpe:
            logger.warning('Max BPE length exceeded: returning None')
            return None

        encoded_sentence = self.auto_tokenizer.encode(sentence_bpes, add_special_tokens=self.cls_start)
        encoded_sentence = np.array(encoded_sentence)

        return encoded_sentence, tokens_offsets

    def get_output_size(self) -> int:
        return self.use_last_n_layers * self.auto_model.config.hidden_size

    def _forward(
            self,
            sentences: torch.Tensor,
            padding_mask: torch.Tensor,
            tokens_offsets: List[List[Tuple[int, int]]],
            sentence_ids: List[str],
            sense_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:

        encoded_sentences = torch.cat(
            self.auto_model(sentences, attention_mask=padding_mask)[2][-self.use_last_n_layers:],
            dim=-1
        )

        batch_size, _, emb_size = encoded_sentences.shape
        max_len = max([len(to) for to in tokens_offsets])

        # build token offsets
        first_tokens_indices = [sentences.new_tensor([_to[0] for _to in sto], dtype=torch.long) for sto in tokens_offsets]
        first_tokens_indices = pad_sequence(first_tokens_indices, batch_first=True, padding_value=-1)
        first_tokens_indices_mask = first_tokens_indices != -1
        first_tokens_indices[~first_tokens_indices_mask] = 0

        # compute vectors
        vector_out = encoded_sentences.gather(dim=1, index=first_tokens_indices.unsqueeze(-1).expand(batch_size, max_len, emb_size))
        vector_out[~first_tokens_indices_mask] = 0

        if sense_mask is not None:
            assert vector_out.shape[1] == sense_mask.shape[1], \
                f"vector_out and sense_mask don't have the same shape ({vector_out.shape}), ({sense_mask.shape})"

        return vector_out
