# Hinglish
Tools and Data

[Data and Model files](https://drive.google.com/drive/folders/12qEbxbefBY24-YqahVV0v7q_IFyxz3L8?usp=sharing)

![Logo](./Hinglish-Logo.png)


Approach | LM Perplexity | Classifier F1 |
---| --- | ---|
BERT|8.2 | 0.63|
DistilBERT|6.5 | 0.63|
ULMFIT | 21 | 0.61|
RoBERTa| 7.54 | 0.64|

## Model Performance

Base LM | Dataset|  Accuracy |  Precision |  Recall |  F1| LM Perplexity|
--|--|--|--|--|--|--|
bert-base-multilingual-cased | Test |  0.686| 0.695| 0.683|  0.685| 8.2|
bert-base-multilingual-cased | Valid |  0.62| 0.592 |  0.605|  0.55| 8.2|
distilbert-base-uncased | Test| 0.671| 0.671| 0.691| 0.677| 6.51|
distilbert-base-uncased | Valid| 0.607| 0.614| 0.600| 0.592| 6.51|
distilbert-base-multilingual-cased | Test| 0.612| 0.615| 0.616| 0.616| 8.1|
distilbert-base-multilingual-cased | Valid| 0.55| 0.531| 0.537| 0.495| 8.1|
roberta-base | Test| 0.630| 0.629| 0.644| 0.635| 7.54|
roberta-base | Valid| 0.60| 0.617| 0.607| 0.595| 7.54|
Ensemble |  Test| 0.713| 0.715| 0.722| 0.718| |
