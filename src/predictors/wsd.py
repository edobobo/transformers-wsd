#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

from typing import List

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides


class WSDPredictor(Predictor):

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, tokens: List[str]) -> JsonDict:
        raise NotImplementedError

    def batch_predict(self, sentences_tokens: List[List[str]]) -> List[JsonDict]:
        raise NotImplementedError

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        raise NotImplementedError


@Predictor.register('wsd')
class SimpleWSDPredictor(WSDPredictor):

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, tokens: List[str]) -> JsonDict:
        return self.predict_json({'tokens': tokens})

    def batch_predict(self, sentences_tokens: List[List[str]]) -> List[JsonDict]:
        return self.predict_batch_json([{'tokens': tokens} for tokens in sentences_tokens])

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = json_dict['tokens']
        return self._dataset_reader.text_to_instance(tokens)
