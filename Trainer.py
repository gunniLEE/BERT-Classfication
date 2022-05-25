import os
import glob
import logging
from tqdm import tqdm, trange
from setproctitle import setproctitle

import numpy as np
import gluonnlp as nlp

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from transformers import BertModel
from transformers.optimization import get_cosine_schedule_with_warmup

from dataloader import BERTDataset, load_tokenizer, load_vocab, KFold_split
from utils import compute_metrics, MODEL_PATH_MAP, calc_accuracy

logger=logging.getLogger(__name__)
setproctitle("nlp_classification_")

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=3,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

class Trainer(object):
    def __init__(self,args, dataset=None, Kfold=None, tok=None):
        try:
            self.args=args
            self.Kfold=Kfold
            self.dataset=dataset
            self.tok=tok

            self.model=BertModel.from_pretrained(args.model_name_or_path)

            # GPU or CPU
            self.device="cuda:0" if torch.cuda.is_available() else "cpu"
            
            logger.info(self.device)
            logger.info(torch.cuda.is_available())

            self.model=BERTClassifier(self.model, dr_rate=0.5).to(self.device)
        except:
            raise Exception("Error occured during Trainer process")

    def train(self):
        n_iter=0
        acc=[]

        for train_index, test_index in self.Kfold.split(self.dataset):
            n_iter+=1

            train=self.dataset.iloc[train_index]
            test=self.dataset.iloc[test_index]

            if not os.path.exists(self.args.kfold_dataset_dir):
                os.mkdir(self.args.kfold_dataset_dir)
                os.chdir(self.args.kfold_dataset_dir)

            train.to_csv(self.args.data_dir_train_filename, sep='\t', index=False)
            test.to_csv(self.args.data_dir_train_filename, sep='\t', index=False)

            dataset_train=nlp.data.TSVDataset(self.args.data_dir_train_filename, field_indices=[1,2], num_discard_samples=1)
            dataset_test=nlp.data.TSVDataset(self.args.data_dir_test_filename, field_indices=[1,2], num_discard_samples=1)

            dataset_train=BERTDataset(dataset_train, 0, 1, self.tok, self.args.max_len, True, False)
            dataset_train=BERTDataset(dataset_test, 0, 1, self.tok, self.args.max_len, True, False)
            
            train_dataloader=torch.utils.data.DataLoader(dataset_train, batch_size=self.args.batch_size, shuffle=True, num_workers=5)
            test_dataloader=torch.utils.data.DataLoader(dataset_test, batch_size=self.args.batch_size, shuffle=True, num_workers=5)

            ## prepare optimizer and schedule
            np_decay=['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                    {'params' : [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
                    {'params' : [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay':0.00}
            ]
            optimizer=Adamw(optimizer_grouped_parameters, lr=self.args.learning_rate)
            loss_fn=nn.CrossEntropyLoss()

            t_total=len(train_dataloader) * self.args.num_epochs
            warmup_step=int(t_total * self.args.warmup_ratio)
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

            for e in range(self.args.num_epochs):
                train_acc = 0.0
                test_acc = 0.0
                self.model.train()
                for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
                    optimizer.zero_grad()
                    token_ids=token_ids.long().to(self.device)
                    segment_ids=segment_ids.long().to(self.device)
                    valid_length=valid_length
                    label=label.long().to(self.device)
                    out=self.model(token_ids, valid_length, segment_ids)
                    loss=loss_fn(out,label)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    train_acc+= calc_accuracy(out,label)
                    if batch_id % self.args.log_interval==0:
                        logger.info("epoch {} batch_id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc/(batch_id+1)))
                logger.info("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
                self.model.eval()
                for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
                    token_ids=token_ids.long().to(self.device)
                    segment_ids=segment_ids.long().to(self.device)
                    valid_length=valid_length
                    label=label.long().to(self.device)
                    out=self.model(token_ids, valid_length, segment_ids)
                    test_acc += calc_accuracy(out,label)
                logger.info("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
                acc.append(test_acc / (batch_id+1))

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.save_model_dir):
            os.makedirs(self.args.save_model_dir)
        
        torch.save(self.args, os.path.join(self.args.save_model_dir, 'training_args.bin'))
        torch.save(self.model.state_dict(), os.path.join(self.args.save_model_dir, 'training_model.bin'))
        logger.info("Saving model checkpoint to %s", self.args.save_model_dir)