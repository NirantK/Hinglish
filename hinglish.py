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
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (AdamW, BertConfig, BertForSequenceClassification,
                          BertTokenizer, DistilBertTokenizer, RobertaTokenizer,
                          get_cosine_with_hard_restarts_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

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
        lm_model_dir :str = None
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
        logger.info(f"Setup model training for {model_name}")
        logger.info(f"---- Parameters for this model ----")
        logger.info(f"Model_name - {self.model_name}")
        logger.info(f"Model_name - {self.model_name}")
        logger.info(f"Batch_size - {self.batch_size}")
        logger.info(f"Attention_probs_dropout_prob - {self.attention_probs_dropout_prob}")
        logger.info(f"Learning_rate - {self.learning_rate}")
        logger.info(f"Adam_epsilon - {self.adam_epsilon}")
        logger.info(f"Hidden_dropout_prob - {self.hidden_dropout_prob}")
        logger.info(f"Epochs - {self.epochs}")
        logger.info("--------------------------------")
        self.device_name = tf.test.gpu_device_name()
        self.device = check_for_gpu(self.device_name, self.model_name)
        self.lm_model_dir = lm_model_dir
        if not lm_model_dir:
            if self.model_name =="bert":
                self.lm_model_dir = "model_save"
            elif self.model_name == "distilbert":
                self.lm_model_dir = "distilbert6"
            elif self.model_name == "roberta":
                self.lm_model_dir = "roberta3"
    def setup(self):
        sentences, labels, le = load_sentences_and_labels()
        tokenizer, input_ids = tokenize_the_sentences(sentences, self.model_name, self.lm_model_dir)
        input_ids, MAX_LEN = add_padding(tokenizer, input_ids, self.model_name)
        attention_masks = create_attention_masks(input_ids)
        (
            train_inputs,
            train_masks,
            train_labels,
            validation_inputs,
            validation_masks,
            validation_labels,
        ) = load_masks_and_inputs(input_ids, labels, attention_masks)
        batch_size, config, learning_rate, adam_epsilon = all_the_important_configs(
            "bert",
            self.batch_size,
            self.attention_probs_dropout_prob,
            self.learning_rate,
            self.adam_epsilon,
            self.hidden_dropout_prob,
        )
        train_dataloader, validation_dataloader = make_dataloaders(
            train_inputs,
            train_masks,
            train_labels,
            batch_size,
            validation_inputs,
            validation_masks,
            validation_labels,
        )
        model = load_lm_model(config, self.model_name, self.lm_model_dir)
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            eps=adam_epsilon,
        )
        total_steps = len(train_dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=total_steps,
        )


def train_model(
    epochs,
    model,
    train_dataloader,
    format_time,
    device,
    optimizer,
    scheduler,
    run_valid,
    loss_values,
    name,
):
    for epoch_i in range(0, epochs):

        logger.info("Training...\n")

        t0 = time.time()

        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            clear_output(wait=True)

            if step % 40 == 0 and not step == 0:
                logger.info(
                    "======== Epoch {:} / {:} ========\n".format(epoch_i + 1, epochs)
                )

                elapsed = format_time(time.time() - t0)

                logger.info(
                    "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.\n".format(
                        step, len(train_dataloader), elapsed
                    )
                )

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

            loss = outputs[0]

            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

        elapsed = format_time(time.time() - t0)
        run_valid()

        avg_train_loss = total_loss / len(train_dataloader)

        loss_values.append(avg_train_loss)

        logger.info("")
        logger.info(
            "  Average training loss: {0:.2f}\n".format(avg_train_loss)
        )
        logger.info(
            "  Training epcoh took: {:}\n".format(format_time(time.time() - t0))
        )

    logger.info("\n")
    logger.info("Training complete!\n")

t = HinglishTrainer("bert")