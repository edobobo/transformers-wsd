#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import torch
from torch import nn


class Swish(nn.Module):
    """

    credits: https://github.com/pytorch/pytorch/pull/3182

    Implementation of Swish: a Self-Gated Activation Function
        Swish activation is simply f(x)=x⋅sigmoid(x)
        Paper: https://arxiv.org/abs/1710.05941
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    """

    def forward(self, input):
        return input * torch.sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'
