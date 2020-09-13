import datetime
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from IPython.display import clear_output
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    AdamW,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from hinglishutils import *


def hinglishDistilBert(
    batch_size=8,
    attention_probs_dropout_prob=0.4,
    learning_rate=5e-7,
    adam_epsilon=1e-8,
    hidden_dropout_prob=0.3,
    input_name="DistilBert",
):
    global name
    name = input_name
    open(f"{name}.log", "w").write(f"------ Starting for model {name}------\n")
    sentences, labels, le = load_sentences_and_labels()
    device_name = tf.test.gpu_device_name()
    device = check_for_gpu(device_name, name)
    tokenizer, input_ids = tokenize_the_sentences(sentences)
    input_ids, MAX_LEN = add_padding(tokenizer, input_ids, name)
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
        "distilbert",
        batch_size,
        attention_probs_dropout_prob,
        learning_rate,
        adam_epsilon,
        hidden_dropout_prob,
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
    model = load_lm_model(config)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        eps=adam_epsilon,
    )
    epochs = 3
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps,
    )

    def run_valid():

        open(f"{name}.log", "a").write("\n")
        open(f"{name}.log", "a").write("Running Validation...\n")

        t0 = time.time()

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        eval_p = 0
        eval_r = 0
        eval_f1 = 0

        (
            eval_accuracy,
            nb_eval_steps,
            eval_p,
            eval_r,
            eval_f1,
        ) = evaluate_data_for_one_epochs(
            eval_accuracy, eval_p, eval_r, eval_f1, nb_eval_steps
        )
        open(f"{name}.log", "a").write(
            "  Accuracy: {0:.2f}\n".format(eval_accuracy / nb_eval_steps)
        )
        open(f"{name}.log", "a").write(
            f"  Precision, Recall F1: {eval_p/nb_eval_steps}, {eval_r/nb_eval_steps}, {eval_f1/nb_eval_steps}\n"
        )
        open(f"{name}.log", "a").write(
            "  Validation took: {:}\n".format(format_time(time.time() - t0))
        )

    def evaluate_data_for_one_epochs(
        eval_accuracy, eval_p, eval_r, eval_f1, nb_eval_steps
    ):
        for batch in validation_dataloader:

            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():

                outputs = model(b_input_ids, attention_mask=b_input_mask)

            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            temp_eval_f1 = flat_prf(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            eval_p += temp_eval_f1[0]
            eval_r += temp_eval_f1[1]
            eval_f1 += temp_eval_f1[2]

            nb_eval_steps += 1
        return eval_accuracy, nb_eval_steps, eval_p, eval_r, eval_f1

    set_seed()

    loss_values = []

    train_model(
        epochs,
        model,
        train_dataloader,
        format_time,
        device,
        optimizer,
        scheduler,
        run_valid,
        loss_values,
    )
    _ = evaluate_final_text(
        tokenizer, MAX_LEN, model, device, le, final_name="test.json", name=name
    )
    full_output = evaluate_final_text(
        tokenizer, MAX_LEN, model, device, le, final_name="final_test.json",name=name
    )

    l = pd.read_csv("test_labels_hinglish.txt")
    precision_recall_fscore_support(
        full_output["Sentiment"], l["Sentiment"][:-1], average="macro"
    )
    open(f"{name}.log", "a").write(
        str(accuracy_score(full_output["Sentiment"], l["Sentiment"][:-1]))
    )

    save_model(full_output, model, tokenizer)


def save_model(full_output, model, tokenizer):
    full_output.to_csv("DistilBertpreds.csv")

    output_dir = f"./{name}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    open(f"{name}.log", "a").write("Saving model to %s\n" % output_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


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
):
    for epoch_i in range(0, epochs):

        open(f"{name}.log", "a").write("\n")

        open(f"{name}.log", "a").write("Training...\n")

        t0 = time.time()

        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            clear_output(wait=True)

            if step % 40 == 0 and not step == 0:
                open(f"{name}.log", "a").write(
                    "======== Epoch {:} / {:} ========\n".format(epoch_i + 1, epochs)
                )

                elapsed = format_time(time.time() - t0)

                open(f"{name}.log", "a").write(
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

        open(f"{name}.log", "a").write("")
        open(f"{name}.log", "a").write(
            "  Average training loss: {0:.2f}\n".format(avg_train_loss)
        )
        open(f"{name}.log", "a").write(
            "  Training epcoh took: {:}\n".format(format_time(time.time() - t0))
        )

    open(f"{name}.log", "a").write("\n")
    open(f"{name}.log", "a").write("Training complete!\n")


def load_lm_model(config):
    model = DistilBertForSequenceClassification.from_pretrained("distilBert6", config=config)
    model.cuda()
    params = list(model.named_parameters())
    return model


def tokenize_the_sentences(sentences):

    open(f"{name}.log", "a").write("Loading DistilBert tokenizer...\n")
    tokenizer = DistilBertTokenizer.from_pretrained("distilBert6")
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    open(f"{name}.log", "a").write("Tokenize the first sentence:\n")
    open(f"{name}.log", "a").write(str(tokenized_texts[0]))
    open(f"{name}.log", "a").write("\n")
    input_ids = []
    for sent in sentences:

        encoded_sent = tokenizer.encode(
            sent,
            add_special_tokens=True,
        )

        input_ids.append(encoded_sent)

    return tokenizer, input_ids
