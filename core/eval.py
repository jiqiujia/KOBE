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

    testCats = ['cloth']#, 'food', 'baby', 'beauty', 'shoe', 'daily', 'luggage', 'appliance', 'jiaju']
    # testCats = ['cloth']
    with io.open("aiProductTest.txt", 'w+', encoding='utf-8') as fout:
        for testCat in testCats:
            with io.open("test.txt".format(testCat),
                    encoding='utf-8') as fin:
                srcList = []
                srcIdList = []
                srcLenList = []

                batch_size = 10
                for line in fin.readlines():
                    line = line.strip().split("\t")[-1]
                    chars = [c for c in line]
                    ids = src_vocab.convertToIdx(chars, dict_helper.UNK_WORD)
                    # print(chars, ids)

                    srcList.append(line)
                    srcIdList.append(ids)
                    srcLenList.append(len(ids))

                # srcList = srcList[:2]
                resList = []
                addOne = 1 if (len(srcList) % batch_size) else 0
                for i in range(len(srcList) // batch_size + addOne):
                    print("batch ", i)
                    startIdx = i * batch_size
                    endIdx = min((i + 1) * batch_size, len(srcList))

                    xs = srcIdList[startIdx:endIdx]
                    maxLen = max(len(x) for x in xs)
                    xs = [x + [0] * (maxLen - len(x)) for x in xs]

                    with torch.no_grad():
                        # samplesBatch, alignmentsBatch = model.beam_sample(torch.tensor(xs), torch.tensor(srcLenList[startIdx:endIdx]),
                        #                                        None, None, beam_size=10, n_best=100)
                        samplesBatch, alignmentsBatch = model.nucleus_sample(torch.tensor(xs), torch.tensor(srcLenList[startIdx:endIdx]),
                                                               None, None, beam_size=20, n_best=10, topk=10, temperature=1)
                        # samples, alignment = model.sample(torch.tensor(xs), torch.tensor(srcLenList[startIdx:endIdx]), None, None)
                        for samples in samplesBatch:
                            candidates = [''.join(tgt_vocab.convertToLabels(s, utils.EOS)) for s in samples]
                            resList.append(candidates)

                for src, res in zip(srcList, resList):
                    fout.write(src + '\n')
                    for desc in res:
                        fout.write("\t"+desc+"\n")
                    fout.write("\n")

                fout.write("\n####################################\n\n")
