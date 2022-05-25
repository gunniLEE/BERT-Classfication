import os
import logging
import random
import torch
import numpy as np

# MODEL_CLASSES = {'bert' : BertModel}
MODEL_PATH_MAP = {'bert' : '/classification/KoBERT/monologg/kobert'}

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M%S',
                        level=logging.info)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_score(preds, labels)

def simple_accuracy(preds, labels):
    return {
        "acc": simple_accuracy(preds, labels),
    }

def softmax(a):
    np.set_printoptions(precision=4)
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X,1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc