from email.mime import base
import os
from posixpath import split

class Trainconfig():
    base_path = "/classification"

    tokenizer_model_name_or_path = os.path.join(base_path, "kobert_news_wiki_ko_cased.spiece")
    split_data = 5
    seed = 777
    vocab_file = os.path.join(base_path, "kobert_news_wiki_ko_cased.spiece")
    model_type = 'bert'
    model_dir = os.path.join(base_path, "/model/kobert")
    kfold_dataset_dir = os.path.join(base_path, "/test_code_dir")
    save_model_dir = os.path.join(base_path, '/test_code/model_file')
    data_dir_train_filename = "train.txt"
    data_dir_test_filename = "test.txt"
    hidden_size = 768
    num_classes = 3
    batch_size = 16
    max_len = 256
    num_worker = 5
    learning_rate = 5e-5
    num_epochs = 5
    warmup_ratio = 0.1
    max_grad_norm = 1
    log_interval = 200

class pred_config():
    base_path = "/classification"

    no_cuda = False
    model_dir = os.path.join(base_path, '/test_code/model_file')
    input_file = "sample_input.txt"
    output_file = "sample_output.txt"
    batch_size = 1
    num_workers= 5
    max_len = 256
    tokenizer_model_name_or_path = os.path.join(base_path, "kobert_news_wiki_ko_cased.spiece")
    vocab_file = os.path.join(base_path, "kobert_news_wiki_ko_cased.spiece")