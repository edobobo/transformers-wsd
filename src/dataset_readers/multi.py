#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import json
import logging
import numpy as np
from typing import Iterable, Optional, List, Tuple

from allennlp.common import Lazy
from allennlp.data import DatasetReader, Instance


logger = logging.getLogger(__name__)


@DatasetReader.register('multi')
class MultiDatasetReader(DatasetReader):

    def __init__(self, dataset_readers: List[Lazy[DatasetReader]], batch_size: int, lazy: bool = False) -> None:
        super().__init__(lazy)
        self.dataset_readers = [dr.construct(lazy=lazy) for dr in dataset_readers]
        self.batch_size = batch_size

    def _read(self, file_path: str) -> Iterable[Instance]:

        try:
            weighted_paths: List[Tuple[str, float]] = json.loads(file_path)
            logger.info('JSON multi-path detected')
        except:
            weighted_paths: List[Tuple[str, float]] = [(file_path, 1.0)]
            logger.info('single-path detected')

        probabilities_weight = sum([t[1] for t in weighted_paths])
        assert probabilities_weight == 1.0, f'Probabilites expected to sum up to 1.0, found {probabilities_weight}'
        assert len(weighted_paths) == len(self.dataset_readers), f'{len(self.dataset_readers)} dataset readers were given, but only {len(weighted_paths)} path was provided'

        # setup data structures
        fps = [fp for fp, _ in weighted_paths]
        fp2p = {fp: p for fp, p in weighted_paths}
        fp2dr = {fp: self.dataset_readers[i] for i, fp in enumerate(fps)}
        fp2it = {fp: dr._read(fp) for fp, dr in fp2dr.items()}
        fp2read_count = {fp: 0 for fp in fps}

        # sampling function
        def sample_file_path() -> str:
            return str(np.random.choice(fps, 1, p=[fp2p[fp] for fp in fps])[0])

        # line reading function
        def read_instance_from_file_path(fp: str) -> Instance:
            try:
                return next(fp2it[fp])
            except StopIteration:
                logger.info(f'{fp} fully read. Restarting it')
                fp2read_count[fp] += 1
                fp2it[fp] = fp2dr[fp]._read(fp)
                return next(fp2it[fp])

        # iterate

        batch: List[Instance] = []
        fp = sample_file_path()

        while any(read_count == 0 for _, read_count in fp2read_count.items()):

            if len(batch) == self.batch_size:
                yield from batch
                batch = []
                fp = sample_file_path()

            batch.append(read_instance_from_file_path(fp))

    def text_to_instance(self, nth_dataset_reader: int, *args, **kwargs) -> Instance:
        # todo implement: how? something like?
        # return self.dataset_readers[nth_dataset_reader].text_to_instance(*args, **kwargs)
        raise NotImplementedError
