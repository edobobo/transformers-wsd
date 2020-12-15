#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import logging
from typing import Iterable, Optional, List

import numpy as np
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import MetadataField, ArrayField

from allen_components.fields import TargetSequenceField
from dataset_readers.utils.multilabel_selection_strategy import MultiLabelSelectionStrategy
from dataset_readers.utils.wsd_instance_conversion_strategy import WSDInstanceConversionStrategy
from model.encoders.base import Encoder
from utils.wsd import expand_raganato_path, read_from_raganato


logger = logging.getLogger(__name__)


@DatasetReader.register('raganato')
class RaganatoDatasetReader(DatasetReader):

    def __init__(
            self,
            encoder: Encoder,
            wsd_instance_conversion_strategy: WSDInstanceConversionStrategy,
            multilabel_selection_strategy: MultiLabelSelectionStrategy,
            limit: int = -1,
            min_length: int = -1,
            max_length: int = -1,
            lazy: bool = False
    ):
        super().__init__(lazy)
        self.encoder = encoder
        self.wsd_instance_conversion_strategy = wsd_instance_conversion_strategy
        self.multilabel_selection_strategy = multilabel_selection_strategy
        self._limit = limit
        self._min_length = min_length
        self._max_length = max_length

    def _read(self, file_path: str) -> Iterable[Instance]:

        # expand raganato path: fails if file_path is not a raganato path
        data_path, key_path = expand_raganato_path(file_path)

        # raganato read
        discarded_due_to_min_length = 0
        discarded_due_to_max_length = 0

        for i, (_, _, sentence) in enumerate(read_from_raganato(data_path, key_path, instance_transform=self.wsd_instance_conversion_strategy.convert)):

            if self._limit != -1 and i > self._limit:
                break

            # extract tokens
            tokens = [instance.annotated_token.text for instance in sentence]

            # extract labels
            labels = []

            for instance in sentence:
                if instance.labels is not None:
                    labels.append(self.multilabel_selection_strategy.select(instance.labels))
                else:
                    labels.append('NOSENSE')

            instance = self.text_to_instance(tokens, sentence_id=f'{file_path}/{i}', labels=labels)
            length = len(instance['sentences'])

            if self._min_length != -1 and length < self._min_length:
                discarded_due_to_min_length += 1
                if discarded_due_to_min_length % 1000 == 0:
                    logger.info(f'{discarded_due_to_min_length} samples have been discarded due to being shorter than min length {self._min_length}')

            if self._max_length != -1 and length > self._max_length:
                discarded_due_to_max_length += 1
                if discarded_due_to_max_length % 1000 == 0:
                    logger.info(f'{discarded_due_to_max_length} samples have been discarded due to being longer than max length {self._max_length}')
            
            yield instance

            # try:
            #     yield self.text_to_instance(tokens, sentence_id=f'{file_path}/{i}', labels=labels)
            # except Exception as e:
            #     continue

        logger.info(f'{discarded_due_to_min_length} samples have been discarded due to being shorter than min length {self._min_length}')
        logger.info(f'{discarded_due_to_max_length} samples have been discarded due to being longer than max length {self._max_length}')

    def text_to_instance(self, tokens: List[str], sentence_id: Optional[str] = None, labels: Optional[List[str]] = None) -> Instance:

        # todo conversion and selection strategy should be used in text_to_instance rather than in read

        # encode
        encoded_tokens, token_offsets = self.encoder.encode_tokens(tokens)

        # build fields dict
        fields = {
            'sentences': ArrayField(encoded_tokens, dtype=np.long),
            'tokens_offsets': MetadataField(token_offsets),
            'padding_mask': ArrayField(np.ones_like(encoded_tokens), dtype=np.long, padding_value=0)
        }

        if sentence_id:
            fields['sentence_ids'] = MetadataField(sentence_id)

        if labels:

            fields["labels"] = TargetSequenceField(labels, namespace='senses')

            # build sense mask
            fields["sense_mask"] = ArrayField(
                np.array([label != 'NOSENSE' for label in labels]),
                padding_value=0,
                dtype=bool
            )

        return Instance(fields)
