#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

from allennlp.common import Registrable
import torch
from torch import nn


class TrainingStrategy(Registrable):

    def __init__(self, module: nn.Module):
        self.module = module
        self._current_step = torch.tensor(0, dtype=torch.long, device=torch.device('cpu'))

    def step(self):
        assert self.module.training, f'Training strategy step called on a non-training step'
        self._current_step += 1
        self._step()

    def _step(self):
        raise NotImplementedError


@TrainingStrategy.register('feature-based')
class FeatureBasedTrainingStrategy(TrainingStrategy):

    def __init__(self, module: nn.Module):
        super().__init__(module)
        for param in self.module.parameters():
            param.requires_grad = False

    def _step(self):
        pass


@TrainingStrategy.register('fine-tuning')
class FineTuningTrainingStrategy(TrainingStrategy):

    def __init__(self, module: nn.Module):
        super().__init__(module)
        for param in self.module.parameters():
            param.requires_grad = True

    def _step(self):
        pass
