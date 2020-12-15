#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

from typing import Dict, Optional, List, Tuple

import torch
from allennlp.common import Lazy
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy

from model.classification_heads import ClassificationHead
from model.encoders.base import Encoder


@Model.register('wsd', constructor='from_partial_objects')
class WSDModel(Model):

    @classmethod
    def from_partial_objects(
            cls,
            vocab: Vocabulary,
            encoder: Encoder,
            classification_head: Lazy[ClassificationHead],
            target_namespace: str
    ):

        # build classification head
        classification_head = classification_head.construct(
            encoder_output_size=encoder.get_output_size(),
            vocab_size=vocab.get_vocab_size(target_namespace)
        )

        return cls(
            vocab,
            encoder,
            classification_head
        )

    def __init__(
            self,
            vocab: Vocabulary,
            encoder: Encoder,
            classification_head: ClassificationHead
    ):

        super().__init__(vocab)

        # architecture
        self.encoder = encoder
        self.classification_head = classification_head

        # metrics
        self.accuracy_1 = CategoricalAccuracy()
        self.accuracy_3 = CategoricalAccuracy(top_k=3)

    def forward(
            self,
            sentences: torch.Tensor,
            tokens_offsets: List[List[Tuple[int, int]]],
            padding_mask: torch.tensor,
            sentence_ids: Optional[List[str]] = None,
            labels: Optional[torch.Tensor] = None,
            sense_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.tensor]:

        encoder_out = self.encoder(sentences, padding_mask, tokens_offsets, sentence_ids, sense_mask)
        classification_out = self.classification_head(encoder_out, labels, sense_mask)

        output = {
            'logits': classification_out.logits,
            'pred_probabilities': classification_out.prediction_probabilities,
            'predictions': classification_out.prediction_probabilities.argmax(dim=-1)
        }

        if labels is not None:
            self.accuracy_1(classification_out.prediction_probabilities, labels, sense_mask)
            self.accuracy_3(classification_out.prediction_probabilities, labels, sense_mask)

        if classification_out.loss is not None:
            output['loss'] = classification_out.loss

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = super().get_metrics(reset)
        metrics['accuracy'] = self.accuracy_1.get_metric(reset)
        metrics['accuracy3'] = self.accuracy_3.get_metric(reset)
        return metrics
