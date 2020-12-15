#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import random
from typing import List, Tuple, Optional

import numpy as np
from allennlp.predictors import Predictor

from utils.file_utils import load_lemmapos2labels
from utils.wsd import AnnotatedToken

import logging

logger = logging.getLogger(__name__)


class Disambiguator:

    def __init__(
            self,
            wsd_predictor: Predictor,
            lemmapos2labels_path: Optional[str]
    ):
        self.wsd_predictor = wsd_predictor
        self.vocab = self.wsd_predictor._model.vocab
        self.lemmapos2labels = load_lemmapos2labels(lemmapos2labels_path) if lemmapos2labels_path is not None else None

    def predict_labels_probabilities(
            self,
            sentences_annotated_tokens: List[List[AnnotatedToken]],
            mask: Optional[List[List[bool]]] = None,
            cut_senses: bool = True,
            n_labels: int = 5
    ) -> List[List[List[Tuple[Optional[str], float]]]]:

        def should_process(i: int, j: int):
            if mask is not None:
                return mask[i][j]
            return True

        sentences_tokens = [[at.text for at in sat] for sat in sentences_annotated_tokens]
        batch_output = self.wsd_predictor.batch_predict(sentences_tokens)

        label_vocabulary = self.vocab.get_token_to_index_vocabulary('senses')

        batch_labels_sorted_prob = []
        for i, (sentence_annotated_tokens, prediction_output) in enumerate(zip(sentences_annotated_tokens, batch_output)):

            sentence_labels_prob = prediction_output['pred_probabilities']
            sentence_labels_sorted_prob = []

            for j, (sat, slp) in enumerate(zip(sentence_annotated_tokens, sentence_labels_prob)):

                if not should_process(i, j):
                    sentence_labels_sorted_prob.append([(None, 0.0)])
                    continue

                # compute list of valid sat_labels

                if cut_senses:

                    if sat.lemma is None or sat.pos is None:
                        logger.warning('Passed an annotated token with either "None"'
                                       ' pos ({}) or "None" lemma ({})'.format(sat.pos, sat.lemma))
                        lemmapos = 'none'
                    else:
                        lemmapos = f'{sat.lemma.lower()}#{sat.pos}'

                    possible_labels = self.lemmapos2labels.get(lemmapos, [])

                    if len(possible_labels) == 0:
                        logger.warning(f'# possible labels was empty: {lemmapos}')
                        sat_labels = []
                    elif len(possible_labels) == 1:
                        sat_labels = [(possible_labels[0], 1.0)]
                    else:
                        sat_labels = []
                        for possible_label in possible_labels:
                            if possible_label in label_vocabulary:
                                sat_labels.append((possible_label, slp[label_vocabulary[possible_label]]))

                    # todo this is conceptually wrong
                    if len(possible_labels) > 1 and len(sat_labels) == 0:
                        sat_labels = [(possible_labels[0], 1.0)]

                else:

                    sat_labels = [(self.vocab.get_token_from_index(_id, 'senses'), _v) for _id, _v in enumerate(slp)]

                # rank them

                if len(sat_labels) == 0:
                    sat_labels = [(None, 1.0)]
                else:
                    n_labels_to_take = min(n_labels, len(sat_labels))
                    slp_sorted_ids = np.array([-sl[1] for sl in sat_labels]).argpartition(kth=n_labels_to_take - 1)[:n_labels_to_take]
                    slp_sorted_ids = sorted(slp_sorted_ids, key=lambda x: -sat_labels[x][1])
                    sat_labels = [sat_labels[slp_id] for slp_id in slp_sorted_ids]

                # append them

                sentence_labels_sorted_prob.append(sat_labels)

            batch_labels_sorted_prob.append(sentence_labels_sorted_prob)

        return batch_labels_sorted_prob

    def predict_labels(
            self,
            sentences_annotated_tokens: List[List[AnnotatedToken]],
            mask: Optional[List[List[bool]]] = None,
            cut_senses: bool = True
    ) -> List[List[Optional[str]]]:
        batch_labels_sorted_prob = self.predict_labels_probabilities(sentences_annotated_tokens, mask=mask, cut_senses=cut_senses, n_labels=5)
        return [[label_prob[0][0] for label_prob in sentence_labels_sorted_prob] for sentence_labels_sorted_prob in batch_labels_sorted_prob]


if __name__ == '__main__':

    import allennlp
    allennlp.common.util.import_module_and_submodules('src')

    from utils.allen import load_predictor
    wsd_predictor = Disambiguator(load_predictor('experiments/mbert-fb-no-cache', predictor='wsd'), 'data/dictionaries/it/inventory.bn.txt')

    out = wsd_predictor.predict_labels([[AnnotatedToken('ciao', 'NONE', 'NONE'), AnnotatedToken('sono', 'VERB', 'essere'), AnnotatedToken('Edo', 'NONE', 'NONE')]], mask=[[False, True, False]], cut_senses=True)
    print(out)
