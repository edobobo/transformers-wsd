#!/bin/bash

# setup env
read -p "Setup env? [y/N] "
if [[ $REPLY =~ ^[Yy]$ ]]
then

    # setup conda
    source ~/miniconda3/etc/profile.d/conda.sh

    # create conda env
    read -p "Enter environment name: " env_name
    conda create -yn $env_name python=3.7
    conda activate $env_name

    # install torch
    read -p "Enter cuda version (e.g. 10.1 or none to avoid installing cuda support): " cuda_version
    if [ $cuda_version == "none" ]; then
      conda install -y pytorch torchvision cpuonly -c pytorch
    else
      conda install -y pytorch torchvision cudatoolkit=$cuda_version -c pytorch
    fi

    # install python requirements
    pip install -r requirements.txt

    # download wordnet
    /usr/bin/env python -c "import nltk; nltk.download('wordnet')"

fi

# download training corpora
read -p "Download training corpora? [y/N] "
if [[ $REPLY =~ ^[Yy]$ ]]
then

  # download standard training corpora from raganato framework
  wget -P data/datasets/ http://lcl.uniroma1.it/wsdeval/data/WSD_Training_Corpora.zip
  unzip data/datasets/WSD_Training_Corpora.zip -d data/datasets/
  mv data/datasets/WSD_Training_Corpora data/datasets/train
  rm data/datasets/WSD_Training_Corpora.zip
  find data/datasets/train/ -mindepth 1 -maxdepth 1 -type f -exec rm '{}' \;

fi

# download evaluation corpora
read -p "Download evaluation corpora? [y/N] "
if [[ $REPLY =~ ^[Yy]$ ]]
then

  # download standard evaluation corpora from raganato framework
  wget -P data/datasets/ http://lcl.uniroma1.it/wsdeval/data/WSD_Unified_Evaluation_Datasets.zip
  unzip data/datasets/WSD_Unified_Evaluation_Datasets.zip -d data/datasets/
  mv data/datasets/WSD_Unified_Evaluation_Datasets data/datasets/eval
  rm data/datasets/WSD_Unified_Evaluation_Datasets.zip
  find data/datasets/eval/ -mindepth 1 -maxdepth 1 -type f -exec rm '{}' \;

  # todo download multilingual corpora
  echo "Automatic download of multilingual corpora is not supported. Download them from https://github.com/SapienzaNLP/mwsd-datasets"

fi

