#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import random
from typing import List

from allennlp.common import Registrable


class MultiLabelSelectionStrategy(Registrable):

    def select(self, labels: List[str]) -> str:
        raise NotImplementedError


@MultiLabelSelectionStrategy.register('always-first')
class AlwaysFirstMultiLabelSelectionStrategy(MultiLabelSelectionStrategy):

    def select(self, labels: List[str]) -> str:
        return labels[0]


@MultiLabelSelectionStrategy.register('random')
class RandomMultiLabelSelectionStrategy(MultiLabelSelectionStrategy):

    def select(self, labels: List[str]) -> str:
        return random.choice(labels)
