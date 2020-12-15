#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import textwrap
from typing import Optional, List, Iterator, Dict

import numpy
import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Field, Vocabulary
from allennlp.data.fields import ArrayField, SequenceField
from overrides import overrides


class TargetSequenceField(Field[torch.Tensor]):

    def __init__(
        self,
        targets: List[str],
        namespace: str = "targets",
    ) -> None:
        self.targets = targets
        self._indexed_targets = None
        self._namespace = namespace

    def __iter__(self) -> Iterator[str]:
        return iter(self.targets)

    def __getitem__(self, idx: int) -> str:
        return self.targets[idx]

    def __len__(self) -> int:
        return len(self.targets)

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for target in self.targets:
            counter[self._namespace][target] += 1

    @overrides
    def index(self, vocab: Vocabulary):
        self._indexed_targets = [
            vocab.get_token_index(target, self._namespace)
            for target in self.targets
        ]

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {"num_tokens": len(self.targets)}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        desired_length = padding_lengths["num_tokens"]
        padded_tags = pad_sequence_to_length(self._indexed_targets, desired_length)
        tensor = torch.LongTensor(padded_tags)
        return tensor

    @overrides
    def empty_field(self) -> "TargetSequenceField":
        # The empty_list here is needed for mypy
        empty_list: List[str] = []
        sequence_label_field = TargetSequenceField(empty_list)
        sequence_label_field._indexed_labels = empty_list
        return sequence_label_field

    def __str__(self) -> str:
        length = len(self.targets)
        formatted_labels = "".join(
            "\t\t" + labels + "\n" for labels in textwrap.wrap(repr(self.targets), 100)
        )
        return (
            f"TargetSequenceField of length {length} with "
            f"labels:\n {formatted_labels} \t\tin namespace: '{self._namespace}'."
        )