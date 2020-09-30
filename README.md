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
bert-base-multilingual-cased | Test |  0.688| 0.698| 0.686|  0.687| 8.2|
bert-base-multilingual-cased | Valid |  0.62| 0.592 |  0.605|  0.55| 8.2|
distilbert-base-uncased | Test| 0.693| 0.694| 0.703| 0.698| 6.51|
distilbert-base-uncased | Valid| 0.607| 0.614| 0.600| 0.592| 6.51|
distilbert-base-multilingual-cased | Test| 0.612| 0.615| 0.616| 0.616| 8.1|
distilbert-base-multilingual-cased | Valid| 0.55| 0.531| 0.537| 0.495| 8.1|
roberta-base | Test| 0.630| 0.629| 0.644| 0.635| 7.54|
roberta-base | Valid| 0.60| 0.617| 0.607| 0.595| 7.54|
Ensemble |  Test| 0.714| 0.718| 0.718| 0.718| |

## Ensemble Performace 

Model | Accuracy  | Precision  | Recall  | F1 | Config | Link to Model and output files |
--|--|--|--|--|--|--|
BERT | 0.68866 | 0.69821 | 0.68608 | 0.6875 | Batch Size - 16<br>Attention Dropout - 0.4<br>Learning Rate - 5e-07<br>Adam epsilon - 1e-08<br>Hidden Dropout Probability - 0.3<br>Epochs - 3 | [BERT](https://drive.google.com/drive/folders/1HAYoWX3zG7XEMaSf74K5dKvaBJdwE6U9?usp=sharing) |
DistilBert | 0.69333 | 0.69496 | 0.70379 | 0.6982 | Batch Size - 16<br>Attention Dropout - 0.6<br>Learning Rate - 3e-05<br>Adam epsilon - 1e-08<br>Hidden Dropout Probability - 0.6<br>Epochs - 3 | [DistilBert](https://drive.google.com/drive/folders/1t_2XqwtRpui5l1prZsmCaArmqzPjPGob?usp=sharing) |
EnsembleBert1 | 0.69233 | 0.70236 | 0.69064 | 0.68952 | Batch Size - 4<br>Attention Dropout - 0.7<br>Learning Rate - 5.01e-05<br>Adam epsilon - 4.79e-05<br>Hidden Dropout Probability - 0.1<br>Epochs - 3 | [EnsembleBert1](https://drive.google.com/drive/folders/1-ais3Y04SWFUYHF4KkUAJMDUEdsfu_GB?usp=sharing) |
EnsembleBert2 | 0.691 | 0.7009 | 0.6889 | 0.68872 | Batch Size - 4<br>Attention Dropout - 0.6<br>Learning Rate - 5.13e-05<br>Adam epsilon - 9.72e-05<br>Hidden Dropout Probability - 0.2<br>Epochs - 3 | [EnsembleBert2](https://drive.google.com/drive/folders/1-rpWWvVruIp_WA0mveU2zHn82fZ5Mcl8?usp=sharing) |
EnsembleDistilBert1 | 0.70166 | 0.70377 | 0.70976 | 0.7061 | Batch Size - 16<br>Attention Dropout - 0.8<br>Learning Rate - 3.02e-05<br>Adam epsilon - 9.35e-05<br>Hidden Dropout Probability - 0.4<br>Epochs - 3 | [EnsembleDistilBert1](https://drive.google.com/drive/folders/1jqcXPLysVSVCOh5ySKa-fMRIWA_djT_P?usp=sharing) |
EnsembleDistilBert2 | 0.689 | 0.691 | 0.69666 | 0.69335 | Batch Size - 4<br>Attention Dropout - 0.6<br>Learning Rate - 5.13e-05<br>Adam epsilon - 9.72e-05<br>Hidden Dropout Probability - 0.2<br>Epochs - 3 | [EnsembleDistilBert2](https://drive.google.com/drive/folders/1-3mwr1v3OBzlSrFxOKpec8ERrqHTPaZo?usp=sharing) |
EnsembleDistilBert3 | 0.69366 | 0.69538 | 0.70557 | 0.69905 | Batch Size - 16<br>Attention Dropout - 0.4<br>Learning Rate - 4.74e-05<br>Adam epsilon - 4.09e-05<br>Hidden Dropout Probability - 0.6<br>Epochs - 3 | [EnsembleDistilBert3](https://drive.google.com/drive/folders/1-KHIKd425T98r0lMjKCv0X7GKaU7K9D5?usp=sharing) |
Ensemble | 0.71466 | 0.71867 | 0.71853 | 0.7182 | NA | [Ensemble](https://drive.google.com/drive/folders/12Iz0xfxszNMkQE8hxO6ajeTBACoKsWUW?usp=sharing) |
