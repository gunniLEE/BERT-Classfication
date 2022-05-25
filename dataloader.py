import os
import logging
import pandas as pd
import numpy as np
import gluonnlp as nlp

import torch
from torch import random
from torch.utils.data import Dataset, dataloader
from sklearn.model_selection import KFold
from utils import set_seed

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


def load_vocab(args):
    vocab=nlp.vocab.BERTVocab.from_sentencepiece(args.vocab_file)
    return vocab

def load_tokenizer(args):
    tokenizer=nlp.vocab.BERTVocab.from_sentencepiece(args.vocab_file)
    return tokenizer

def tok(tokenizer, vocab):
    return nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def KFold_split(args):
    file=pd.read_csv(args.file_path, sep='\t')
    Kfold=KFold(n_splits=args.split_data, shuffle=True, random_state=args.seed)
    return Kfold, file