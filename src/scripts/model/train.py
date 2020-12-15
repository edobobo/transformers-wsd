#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import os
import random
import tempfile
import warnings
warnings.simplefilter("ignore")

import logging
import sys
import argparse
import json
import collections

from tqdm import tqdm
from typing import List

from allennlp.commands import main as allen_main
from utils.wsd import RaganatoBuilder, read_from_raganato, expand_raganato_path

logging.basicConfig(level=logging.INFO)


def custom_allen_main(command: str, config_file_path: str, serialization_dir: str, overrides: str, recover: bool):

    sys.argv = [
        'allennlp',  # command name, not used by main
        command,
        config_file_path,
        '-s', serialization_dir,
        '--include-package', 'src',
        '-o', overrides,
    ]

    if recover:
        sys.argv.append('-r')

    allen_main()


def split_training_set(training_set_path: str, dev_size: int, output_folder: str, split_seed: int) -> List[str]:
    
    # load training set and shuffle it
    training_set = list(tqdm(read_from_raganato(*expand_raganato_path(training_set_path)), desc='Reading training set'))
    random.Random(split_seed).shuffle(training_set)

    # split it
    datasets = [
        ('train', training_set[dev_size: ]),
        ('dev', training_set[: dev_size])
    ]

    paths = []
    
    # write them
    for name, dataset in datasets:
    
        os.mkdir(f'{output_folder}/{name}')
        output_path = f'{output_folder}/{name}/{name}'

        rb = RaganatoBuilder()
        last_did = None
                
        for did, sid, tokens in tqdm(dataset, desc=f'Writing dataset {name}'):
            if last_did is None or last_did != did:
                last_did = did
                rb.open_text_section(did)
            rb.open_sentence_section(sid.split('.')[-1])
            for token in tokens:
                rb.add_annotated_token(
                    token=token.annotated_token.text,
                    lemma=token.annotated_token.lemma,
                    pos=token.annotated_token.pos,
                    instance_id=token.instance_id.split('.')[-1] if token.instance_id is not None else None,
                    labels=token.labels)

        rb.store(*expand_raganato_path(output_path), prettify=False)
        paths.append(output_path)

    return paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file_path')
    parser.add_argument('model_name')
    parser.add_argument('--train', type=str, default=None, help='Training path')
    parser.add_argument('--dev', type=str, default=None, help='Development path')
    parser.add_argument('--dev-size', type=int, default=None, help='Validation size (in sentences)')
    parser.add_argument('--split-seed', default=1213, help='Split seed used to have the same dev set over multiple runs over the same training set. Default is 1213)')
    parser.add_argument('--recover', action='store_true', default=False)
    return parser.parse_args()


def main():

    args = parse_args()
    serialization_dir = f'experiments/{args.model_name}'

    overrides = collections.defaultdict(dict)

    # process training/dev set if provided
    tmp_data_dir = None
    if args.train is not None and args.dev_size is not None:
        # reserve dev_size sentences from training set
        tmp_data_dir = tempfile.TemporaryDirectory()
        dataset_paths = split_training_set(args.train, args.dev_size, tmp_data_dir.name, split_seed=args.split_seed)
        overrides['train_data_path'] = dataset_paths[0]
        overrides['validation_data_path'] = dataset_paths[1]
    else:
        if args.train is not None:
            overrides['train_data_path'] = args.train
        if args.dev is not None:
            overrides['validation_data_path'] = args.dev

    # set env level variables
    os.environ['wandb_run_name'] = args.model_name

    # delete run experiments if it already exists and if it is a debug run
    if os.path.exists(serialization_dir) and 'debug' in serialization_dir[serialization_dir.rindex('/') + 1:]:
        os.system(f'rm -rf {serialization_dir}')

    # main call
    custom_allen_main(
        'train',
        args.config_file_path,
        serialization_dir,
        json.dumps(overrides),
        args.recover
    )

    if tmp_data_dir is not None:
        tmp_data_dir.cleanup()


if __name__ == '__main__':
    main()
