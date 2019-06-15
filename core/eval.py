import codecs
import json
import os
import pickle
import random
import time
from argparse import Namespace
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import yaml
from torch.nn.init import xavier_uniform_
from tqdm import tqdm

import lr_scheduler as L
import models
import opts
import utils
from dataset import load_data
from utils import misc_utils

import io
import train
import dict_helper

if __name__ == "__main__":
    opt = opts.model_opts()
    config = yaml.load(open(opt.config, "r"))
    config = Namespace(**config, **vars(opt))

    print("loading checkpoint...\n")
    checkpoints = torch.load("E:/dlprojects/data/kobe/experiments/baseline_20190613/best_bleu_checkpoint.pt",
                             map_location=lambda storage, loc: storage)

    src_vocab = dict_helper.Dict(data=os.path.join(config.data, 'src.dict'))
    tgt_vocab = dict_helper.Dict(data=os.path.join(config.data, 'tgt.dict'))
    config.src_vocab_size = src_vocab.size()
    config.tgt_vocab_size = tgt_vocab.size()

    model, optim = train.build_model(checkpoints, config, 'cpu')
    model.eval()

    with io.open("E:/projects/AiProductDescWriter/server_data/cloth/testdata/JDTestTitles.txt", encoding='utf-8') as fin:
        srcList = []
        srcLenList = []
        batch_size = 20
        for line in fin.readlines():
            chars = [c for c in line]
            ids = src_vocab.convertToIdx(chars, dict_helper.UNK_WORD)
            #print(chars, ids)

            with torch.no_grad():
                samples, alignment = model.beam_sample(torch.LongTensor([ids]), torch.LongTensor([len(ids)]), None, None, beam_size=10)

            candidate = [''.join(tgt_vocab.convertToLabels(s, utils.EOS)) for s in samples]

            print(line, ''.join(candidate))