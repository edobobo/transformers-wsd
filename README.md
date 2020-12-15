# Transformers WSD

This repo hosts the classification library we used in:
```
@inproceedings{ijcai2020-531,
  title     = {Mu{L}a{N}: Multilingual Label propagatio{N} for Word Sense Disambiguation},
  author    = {Barba, Edoardo and Procopio, Luigi and Campolungo, Niccolò and Pasini, Tommaso and Navigli, Roberto},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {3837--3844},
  year      = {2020},
  month     = {7},
  doi       = {10.24963/ijcai.2020/531},
  url       = {https://doi.org/10.24963/ijcai.2020/531},
}
```

We are in the process of updating and improving several parts of this framework!

## Setup Local Env

```
bash setup.sh
```

### Multilingual Evaluation Datasets and Dictionaries

See this [repo](https://github.com/SapienzaNLP/mwsd-datasets).

## Train

```
PYTHONPATH=src/ python src/scripts/model/train.py configurations/mbert/feature-based.jsonnet semcor-zero-shot --train data/datasets/train/SemCor/semcor --dev data/datasets/eval/semeval2007/semeval2007
```

### Pre-trained models

We will soon release additional models, especially the MuLaN versions of the paper; for the time being, we release:
* [semcor-mbert-zero-shot](https://drive.google.com/file/d/1fOLdi482xklIar31VZcnGngjOc1N8EMX/view?usp=sharing), corresponding to ∅-shot-SemCor in Table 2 of the paper

## Evaluate

```
PYTHONPATH=src/ python src/scripts/eval/raganato_eval.py experiments/semcor-zero-shot/best.th multilingual-semeval13 --cut-senses --cuda-device 0 --batch-size 16
```

# Contacts
For any question either open an issue on github or contact:
barba\[at\]di\[dot\]uniroma1\[dot\]it (Edoardo Barba)
procopio\[at\]di\[dot\]uniroma1\[dot\]it (Luigi Procopio)

# License
All data and codes provided in this repository are subject to the  Attribution-Non Commercial-ShareAlike 4.0 International license (CC BY-NC 4.0).

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
