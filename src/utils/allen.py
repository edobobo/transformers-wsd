#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import logging
import os

from allennlp.models import load_archive
from allennlp.predictors import Predictor


logger = logging.getLogger(__name__)


def load_predictor(model_path: str, predictor: str, cuda_device: int = -1) -> Predictor:

    weights_file = None

    if not os.path.isdir(model_path) and not model_path.endswith('.tar.gz'):
        weights_file = model_path
        model_path = model_path[: model_path.rindex('/')]
        logger.info('Inferring weights file from provided model path')

    return Predictor.from_archive(
        load_archive(
            model_path,
            weights_file=weights_file,
            cuda_device=cuda_device,
        ),
        predictor
    )
