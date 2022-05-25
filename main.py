import argparse
import logging
from ast import parse

from matplotlib.transforms import Transform

import gluonnlp as nlp
from Trainer import Trainer
from utils import init_logger, set_seed, MODEL_PATH_MAP

from dataloader import BERTDataset, KFold_split, load_tokenizer, load_vocab, tok
from config import TrainConfig

import glob

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--split_data", default=TrainConfig.split_data , type=int, help="split data for Kfold validation")
    parser.add_argument("--file_path", type=str, help="dataset file")
    parser.add_argument("--tokenizer_model_name_or_path", default=TrainConfig.tokenizer_model_name_or_path, type=str, help="tokenizer file path")
    parser.add_argument("--seed", default=TrainConfig.seed, type=int, help="setting random seed")
    parser.add_argument("--hidden_size", default=TrainConfig.hidden_size, type=int, help="hidden size")
    parser.add_argument("--num_classes" default=TrainConfig.num_classes, type=int, help="num classes")
    parser.add_argument("--vocab_file", default=TrainConfig.vocab_file, tpye=str, help="vocab file path")
    parser.add_argument("--model_type", default=TrainConfig.model_type, type=str, help="model type")
    parser.add_argument("--model_dir", default=TrainConfig.model_dir, type=str, help="model_path")
    parser.add_argument("--kfold_dataset_dir", default=TrainConfig.kfold_dataset_dir, type=str, help="make directory")
    parser.add_argument("--data_dir_train_filename", default=TrainConfig.data_dir_train_filename, type=str, help="train data after split")
    parser.add_argument("--data_dir_test_filename", default=TrainConfig.data_dir_test_filename, type=str, help="test data after split")
    parser.add_argument("--batch_size", default=TrainConfig.batch_size, type=int, help="batch size for training")
    parser.add_argument("--max_len", default=TrainConfig.max_len, type=int, help="setting the max length")
    parser.add_argument("--num_workers", default=TrainConfig.num_workers, type=int, help="num workers")
    parser.add_argument("--warmup_ratio", default=TrainConfig.warmup_ratio, type=float, help="warmup_ratio")
    parser.add_argument("--max_grad_norm", default=TrainConfig.max_grad_norm, type=int, help="max_grad_norm")
    parser.add_argument("--log_interval", default=TrainConfig.log_interval, type=int, help="check the accuracy (num interval) ")
    parser.add_argument("--save_model_dir", default=TrainConfig.save_model_dir, type=str, help="model directory after train")

    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]

    init_logger()
    set_seed(args)

    tokenizer=load_tokenizer(args)
    vocab=load_vocab(args)
    tok=tok(tokenizer, vocab)

    Kfold, dataset = KFold_split(args) # args.tokenizer_model_name_or_path
    Trainer=Trainer(args, dataset, Kfold, tok)

    if args.do_train:
        Trainer.train()
        Trainer.save_model()
    else:
        logging.warning("************** TRY AGAIN **************")

