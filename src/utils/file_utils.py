#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import subprocess


def count_lines_in_file(path):
    return int(subprocess.check_output(f"wc -l \"{path}\"", shell=True).split()[0])


_pos_mapping = {'n': 'NOUN', 'a': 'ADJ', 'r': 'ADV', 'v': 'VERB'}


def load_lemmapos2labels(file_path: str) -> dict:
    lemmapos2labels = dict()
    with open(file_path) as f:
        for line in f:
            lemmapos, *synsets = line.strip().split('\t')
            lemma, pos = lemmapos.split('#')
            pos = _pos_mapping.get(pos, pos)
            lemmapos = f'{lemma.lower()}#{pos}'
            # pos = 'r' if pos == 'ADV' else pos.lower()[0]
            # lemmapos = f'{lemma}#{pos}'
            # synsets = parts[1:]
            lemmapos2labels[lemmapos] = synsets
    return lemmapos2labels
