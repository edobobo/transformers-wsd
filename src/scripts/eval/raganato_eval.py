#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)

import argparse
import json
import os
from typing import Optional, Dict, List, NamedTuple, cast, Tuple, Set, TextIO

import allennlp
from prettytable import PrettyTable
from tqdm import tqdm

from dataset_readers.utils.wsd_instance_conversion_strategy import WSDInstanceConversionStrategy, IdentityWSDInstanceConversionStrategy
from utils.allen import load_predictor
from utils.file_utils import load_lemmapos2labels
from utils.wsd import AnnotatedToken, expand_raganato_path, read_from_raganato
from wsd.disambiguator import Disambiguator

logger = logging.getLogger(__name__)


def load_eval_dataset(
        eval_dataset_prefix: str,
        wsd_instance_conversion_strategy: Optional[WSDInstanceConversionStrategy] = None
) -> Tuple[List[List[AnnotatedToken]], List[List[Optional[Tuple[str, Set[str]]]]]]:

    if wsd_instance_conversion_strategy is None:
        wsd_instance_conversion_strategy = IdentityWSDInstanceConversionStrategy()

    eval_dataset_data_path, eval_dataset_key_path = expand_raganato_path(f'{eval_dataset_prefix}')
    raganato_dataset = read_from_raganato(eval_dataset_data_path, eval_dataset_key_path)

    dataset_annotated_sentences = []
    dataset_gold_labels = []

    for _, _, raganato_sentence in raganato_dataset:
        
        raganato_sentence = [wsd_instance_conversion_strategy.convert(e) for e in raganato_sentence]
        sentence_annotated_tokens = [wsd_instance.annotated_token for wsd_instance in raganato_sentence]

        if any([at.text is None for at in sentence_annotated_tokens]):
            continue

        dataset_annotated_sentences.append(sentence_annotated_tokens)
        dataset_gold_labels.append(
            [(wsd_instance.instance_id, set(wsd_instance.labels)) if wsd_instance.labels is not None else None for wsd_instance in raganato_sentence])

    return dataset_annotated_sentences, dataset_gold_labels


def _batch_predict(disambiguator: Disambiguator, dataset_annotated_sentences: List[List[AnnotatedToken]], mask: Optional[List[List[bool]]] = None, batch_size: int = 16, cut_senses: bool = True) -> List[List[Optional[str]]]:

    dataset_predicted_labels = []
    pbar = tqdm(total=len(dataset_annotated_sentences))

    for i in range(0, len(dataset_annotated_sentences), batch_size):
        batch_annotated_sentences = dataset_annotated_sentences[i: i+batch_size]
        batch_mask = mask[i: i+batch_size]
        batch_predicted_labels = disambiguator.predict_labels(batch_annotated_sentences, batch_mask, cut_senses)
        dataset_predicted_labels += batch_predicted_labels
        pbar.update(len(batch_annotated_sentences))

    pbar.close()

    return dataset_predicted_labels


def produce_gs_predictions(disambiguator: Disambiguator, eval_dataset_prefix: str, batch_size: int, cut_senses: bool, wsd_instance_conversion_strategy: WSDInstanceConversionStrategy) -> Tuple:

    dataset_annotated_sentences, dataset_gold_labels = load_eval_dataset(eval_dataset_prefix, wsd_instance_conversion_strategy)
    mask = [[_l is not None for _l in l] for l in dataset_gold_labels]
    dataset_predicted_labels = _batch_predict(disambiguator, dataset_annotated_sentences, mask=mask, batch_size=batch_size, cut_senses=cut_senses)

    ids = dict()
    gs = dict()
    predictions = dict()
    n_instances = 0

    for i, (sentence_predicted_labels, sentence_gold_labels) in enumerate(zip(dataset_predicted_labels, dataset_gold_labels)):

        assert len(sentence_predicted_labels) == len(sentence_gold_labels), \
            f'"Sentence predicted labels" are not of the same size of "sentence gold labels": {len(sentence_predicted_labels)}, {len(sentence_gold_labels)}'

        for j, (spl, sgl) in enumerate(zip(sentence_predicted_labels, sentence_gold_labels)):

            if sgl is None:
                continue

            sid, sgl = sgl

            n_instances += 1
            ids[(i, j)] = sid 
            gs[(i, j)] = sgl

            if spl is None:
                continue

            predictions[(i, j)] = spl

    return ids, gs, predictions, n_instances


class ScoresBag(NamedTuple):
    precision: float
    recall: float
    f1: float
    dataset_size: int
    total_predictions: int
    ok: int
    notok: int


def _compute_scores(Disambiguator: Disambiguator, eval_dataset_path: str, batch_size: int, cut_senses: bool, wsd_instance_conversion_strategy: WSDInstanceConversionStrategy, f_dump_system_response: Optional[TextIO]) -> ScoresBag:

    ids, gs, predictions, n_instances = produce_gs_predictions(Disambiguator, eval_dataset_path, batch_size, cut_senses, wsd_instance_conversion_strategy)

    ok, notok = 0, 0

    for pred_key, pred_synset in predictions.items():

        if pred_key not in gs:
            continue

        gs_synsets = gs[pred_key]

        if pred_synset in gs_synsets:
            ok += 1
        else:
            notok += 1

        if f_dump_system_response is not None:
            f_dump_system_response.write(f'{eval_dataset_path.split("/")[-1]}.{ids[pred_key]}\t{pred_synset}\t{",".join(gs_synsets)}\n')

    precision = ok / (ok + notok)
    recall = ok / len(gs.keys())
    f1 = 0.0 if precision == recall == 0.0 else (2 * precision * recall) / (precision + recall)

    return ScoresBag(precision, recall, f1, n_instances, len(predictions), ok, notok)


class ScoresHandler:

    def __init__(self, config_file_path: str = 'configurations/evaluation/scores_handler.json'):

        with open(config_file_path) as f:
            config = json.load(f)

        self.special_datasets = config.get('special-datasets', dict())

    def compute_scores(self, model_path: str, eval_dataset_path: str, language_id: Optional[str], cuda_device: int, batch_size: int, cut_senses: bool, wsd_instance_conversion_strategy: WSDInstanceConversionStrategy, dump_system_response: Optional[str]) -> Dict[str, ScoresBag]:

        if language_id is None and eval_dataset_path not in self.special_datasets:
            raise ValueError('if the eval_dataset is not a special one the language_id must be always not None')

        wsd_predictor = Disambiguator(
            load_predictor(model_path, predictor='wsd', cuda_device=cuda_device),
            lemmapos2labels_path=None
        )

        language2eval_dataset_path = self.special_datasets.get(eval_dataset_path, [(language_id, eval_dataset_path)])
        
        f_dump_system_response = None
        if dump_system_response is not None:
            f_dump_system_response = open(dump_system_response, 'w')

        scores = dict()
        for _lid, _edp in language2eval_dataset_path:
            print(f'Computing scores for dataset: {_edp}, language_id: {_lid}')
            # todo hardcoded coupling towards babelnet as inventory
            wsd_predictor.lemmapos2labels = load_lemmapos2labels(f'data/dictionaries/{_lid}/inventory.bn.txt')
            scores[_edp] = _compute_scores(wsd_predictor, _edp, batch_size, cut_senses, wsd_instance_conversion_strategy, f_dump_system_response)

        if f_dump_system_response is not None:
            f_dump_system_response.close()

        return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='allennlp model path')
    parser.add_argument('eval_dataset_path', help='path to the evaluation dataset or a special identifier (semeval, semeval13,...')
    parser.add_argument('--language-id', help='language identifier for the model (it, de, ...)')
    parser.add_argument('--cut-senses', action='store_true')
    parser.add_argument('--conversion-strategy', type=str, default='identity', choices=WSDInstanceConversionStrategy.list_available())
    parser.add_argument("--cuda-device", type=int, default=-1, help='Cuda device')
    parser.add_argument("--batch-size", type=int, default=32, help='Batch size to use')
    parser.add_argument('--dump-system-response', type=str, default=None, help='If given, the system will dump the system response at the specified file (format: <instance id> \\t <predicted label> \\t <gold labels>)')
    return parser.parse_args()


def main():

    args = parse_args()

    allennlp.common.util.import_module_and_submodules('src')

    print(f'Model: {args.model_path}, Eval dataset: {args.eval_dataset_path}, Language id: {args.language_id}')

    scores_handler = ScoresHandler()
    wsd_instance_conversion_strategy = WSDInstanceConversionStrategy.by_name(args.conversion_strategy)()

    scores = scores_handler.compute_scores(args.model_path, args.eval_dataset_path, args.language_id, args.cuda_device, args.batch_size, args.cut_senses, wsd_instance_conversion_strategy, args.dump_system_response)

    pt = PrettyTable(['dataset-name', 'dataset-size', 'total-predictions', 'ok', 'not-ok', 'precision', 'recall', 'f1'])
    for dataset, score in scores.items():
        dataset_name = os.path.basename(dataset)
        pt.add_row(
            [dataset_name, score.dataset_size, score.total_predictions,
             score.ok, score.notok,
             '{:.2f}'.format(score.precision * 100),
             '{:.2f}'.format(score.recall * 100),
             '{:.2f}'.format(score.f1 * 100)]
        )

    print(pt)


if __name__ == '__main__':
    main()

