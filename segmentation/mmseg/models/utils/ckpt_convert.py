# This script consists of several convert functions which
# can modify the weights of model in original repo to be
# pre-trained weights.

from collections import OrderedDict

import torch

def smt_convert(ckpt):
    new_ckpt = OrderedDict()
    for i, (k, v) in enumerate(ckpt.items()):
        new_k = k
        new_v = v
        new_ckpt[new_k] = new_v

    del new_ckpt['head.weight']
    del new_ckpt['head.bias']

    return new_ckpt