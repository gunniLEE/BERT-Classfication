## Korean BERT pre-trained cased (KoBERT)

### Why'?'

* 구글 [BERT base multilingual cased](https://github.com/google-research/bert/blob/master/multilingual.md)의 한국어 성능 한계

### Training Environment

* Architecture

```python
predefined_args = {
        'attention_cell': 'multi_head',
        'num_layers': 12,
        'units': 768,
        'hidden_size': 3072,
        'max_length': 512,
        'num_heads': 12,
        'scaled': True,
        'dropout': 0.1,
        'use_residual': True,
        'embed_size': 768,
        'embed_dropout': 0.1,
        'token_type_vocab_size': 2,
        'word_embed': None,
    }
```

### model file
``` shell
git clone https://huggingface.co/monologg/kobert
```
## BERT-Classification

### Train
``` python
python main.py
```
### 분류 개수 조절

Trainer.py 
``` python
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=3,   ## 해당 부분 수정 하여 클래스 수 조정 ##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
```

### Predict
``` python
python predict.py
```
### input
sample_input.txt 파일에 분석할 문장을 입력합니다.

### output
sample_output.txt 파일에 분류 결과가 출력됩니다.
