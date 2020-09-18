import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from IPython.display import clear_output
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertTokenizer,
    RobertaTokenizer,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from hinglishutils import *
from datetime import datetime

logger = logging.getLogger("hinglish")
logger.setLevel(logging.DEBUG)


class HinglishTrainer:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        attention_probs_dropout_prob: float = 0.4,
        learning_rate: float = 5e-7,
        adam_epsilon: float = 1e-8,
        hidden_dropout_prob: float = 0.3,
        epochs: int = 3,
        lm_model_dir: str = None,
    ):
        self.model_name = model_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.hidden_dropout_prob = hidden_dropout_prob
        self.epochs = epochs

        self.timestamp = str(datetime.timestamp(datetime.now()))
        fh = logging.FileHandler(f"{self.model_name}_{self.timestamp}.log")
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.info(f"Setup self.model training for {model_name}")
        logger.info(f"---- Parameters for this self.model ----")
        logger.info(f"Model_name - {self.model_name}")
        logger.info(f"Model_name - {self.model_name}")
        logger.info(f"Batch_size - {self.batch_size}")
        logger.info(
            f"Attention_probs_dropout_prob - {self.attention_probs_dropout_prob}"
        )
        logger.info(f"Learning_rate - {self.learning_rate}")
        logger.info(f"Adam_epsilon - {self.adam_epsilon}")
        logger.info(f"Hidden_dropout_prob - {self.hidden_dropout_prob}")
        logger.info(f"Epochs - {self.epochs}")
        logger.info("--------------------------------")
        self.device_name = tf.test.gpu_device_name()
        self.device = check_for_gpu(self.device_name, self.model_name)
        self.lm_model_dir = lm_model_dir
        if not lm_model_dir:
            if self.model_name == "bert":
                self.lm_model_dir = "model_save"
            elif self.model_name == "distilbert":
                self.lm_model_dir = "distilbert6"
            elif self.model_name == "roberta":
                self.lm_model_dir = "roberta3"

    def setup(self):
        sentences, labels, self.le = load_sentences_and_labels()
        self.tokenizer, input_ids = tokenize_the_sentences(
            sentences, self.model_name, self.lm_model_dir
        )
        input_ids, self.MAX_LEN = add_padding(
            self.tokenizer, input_ids, self.model_name
        )
        attention_masks = create_attention_masks(input_ids)
        (
            train_inputs,
            train_masks,
            train_labels,
            validation_inputs,
            validation_masks,
            validation_labels,
        ) = load_masks_and_inputs(input_ids, labels, attention_masks)
        self.config = all_the_important_configs(
            "bert",
            self.batch_size,
            self.attention_probs_dropout_prob,
            self.learning_rate,
            self.adam_epsilon,
            self.hidden_dropout_prob,
        )
        self.train_dataloader, self.validation_dataloader = make_dataloaders(
            train_inputs,
            train_masks,
            train_labels,
            self.batch_size,
            validation_inputs,
            validation_masks,
            validation_labels,
        )

    def train(self):
        self.setup()
        self.model = load_lm_model(self.config, self.model_name, self.lm_model_dir)
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=self.adam_epsilon,
        )
        total_steps = len(self.train_dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=total_steps,
        )
        set_seed()
        loss_values = []
        train_model(
            self.epochs,
            self.model,
            self.train_dataloader,
            format_time,
            self.device,
            optimizer,
            scheduler,
            run_valid,
            loss_values,
            self.model_name,
        )

    def evaluate(
        self,
        dev_json="test.json",
        test_json="final_test.json",
        test_labels="test_labels_hinglish.txt",
    ):
        _ = evaluate_final_text(
            self.tokenizer,
            self.MAX_LEN,
            self.model,
            self.device,
            self.le,
            final_name=dev_json,
            name=self.model_name,
        )
        full_output = evaluate_final_text(
            self.tokenizer,
            self.MAX_LEN,
            self.model,
            self.device,
            self.le,
            final_name=test_json,
            name=self.model_name,
        )
        l = pd.read_csv(test_labels)
        precision_recall_fscore_support(
            full_output["Sentiment"], l["Sentiment"][:-1], average="macro"
        )
        logger.info(str(accuracy_score(full_output["Sentiment"], l["Sentiment"][:-1])))
        save_model(full_output, self.model, self.tokenizer, self.model_name)
