#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import logging
import re
from typing import Dict, Any, List

import torch
import wandb
from allennlp.data.dataloader import TensorDict
from allennlp.predictors import Predictor
from allennlp.training import EpochCallback, GradientDescentTrainer, BatchCallback


logger = logging.getLogger(__name__)


@EpochCallback.register('wandb')
class WandBCallback(EpochCallback):

    def __init__(self, project_name: str, run_name: str, blacklist_regex: List[str] = ['_MB', '_duration', 'reg.*loss', '_scale']):

        self.project_name = project_name
        self.run_name = run_name

        if 'debug' not in self.run_name:
            wandb.init(project=project_name, name=run_name)
            self.wandb = wandb
            self._blacklist_regex = [re.compile(p) for p in blacklist_regex]
        else:
            logger.info('wandb callback ignored: \'debug\' found in run name')

    def __call__(self, trainer: GradientDescentTrainer, metrics: Dict[str, Any], epoch: int, is_master: bool) -> None:

        if epoch < 0:
            return

        if 'debug' in self.run_name:
            logger.info('wandb callback ignore: \'debug\' found in run name')
            return

        metrics = {k: v for k, v in metrics.items() if type(v) in {float, int} and all(rb.search(k) is None for rb in self._blacklist_regex)}
        metrics['lr'] = trainer.optimizer.param_groups[0]["lr"]
        self.wandb.log(metrics)

