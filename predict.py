import enum
from imp import load_module
from lib2to3.pgen2 import token
from mimetypes import init
import os
from socket import getservbyname
import time
import glob
import logging
import argparse
from regex import F
from tqdm import tqdm, trange

import pandas as pd
import numpy as np
import torch
import gluonnlp as nlp

from torch.utils.data import TensorDataset, dataloader
from transformers import BertModel

from utils import init_logger, softmax, calc_accuracy
from Trainer import BERTClassifier
from dataloader import BERTDataset, load_vocab, load_tokenizer, tok
from config import pred_config

logger=logging.getLogger(__name__)

def get_device(pred_config):
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, "training_args.bin"))

def get_training_model(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'training_model.bin'))

def load_model(pred_config, args, device):
    # check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model dosen't exists! Train First")

    try:
        model=BertModel.from_pretrained(args.model_name_or_path)
        model=BERTClassifier(model, dr_rate=0.5).to(device)
        model.eval()
        logger.info("*********** Model Loaded ***********")
    except:
        raise Exception("Some Model files might be missing...")

    return model


def getSentimentValue(pred_config, model, sample_input, tok, device):
    start=time.time()

    commentlist = []  # 텍스트 데이터 담을 리스트
    pred_result = []  # 결과값 담을 리스트

    for c in sample_input:
        commentlist.append([c,1])
    
    pdData = pd.DataFrame(commentlist, columns=['sentence', 'label'])
    pdData = pdData.values

    test_set = BERTDataset(pdData, 0, 1, tok, pred_config.max_len, True, False)
    test_input = torch.utils.data.DataLoader(test_set, batch_size = pred_config.batch_size, num_workers= pred_config.num_workers)

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_input)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        out = model(token_ids, valid_length, segment_ids)
        prediction = out.cpu().detach().numpy().argmax()
        soft_pred = softmax(out.cpu().detach().numpy())
        pred_result.append([[commentlist[batch_id][0], prediction]])
        end = time.time()
        result=end-start
    
    return pred_result

def predict(pred_config):
    # Load model and args
    args = get_args(pred_config) # --> training args load
    tr_model = get_training_model(pred_config) # --> training parameter load

    device = get_device(pred_config)

    logger.info(device)

    tokenizer = load_tokenizer(args)
    vocab = load_vocab(args)

    global tok

    tok = tok(tokenizer, vocab)

    model = load_model(pred_config, args, device)
    model.load_state_dict(tr_model)

    logger.info("********* Train Model Loaded *********")
    logger.info(model)

    sample_input = []

    # sample input file for inference
    with open(pred_config.input_file, "r", encoding='utf-8') as f:
        for line in f:
            sample_input.append(line.strip())
    
    pred_result = getSentimentValue(pred_config, model, sample_input, tok, device)

    # Write to output file
    with open(pred_config.output_file, 'w', encoding='utf-8') as f:
        for pred in pred_result:
            f.write("{0}\t{1}".format(pred[0][0], pred[0][1]))

    logger.info("********** Prediction Done !!! **********")


if __name__=="__main__":

    init_logger()

    parser=argparse.ArgumentParser()

    parser.add_argument("--input_file", default=pred_config.input_file, help="input file for inference")
    parser.add_argument("--output_file", default=pred_config.output_file, help="result file")
    parser.add_argument("--model_dir", default=pred_config.model_dir, help = "model file directory")

    predict(pred_config)



