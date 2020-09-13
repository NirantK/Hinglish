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
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def all_the_important_configs(
    batch_size=8,
    attention_probs_dropout_prob=0.4,
    learning_rate=5e-7,
    adam_epsilon=1e-8,
    hidden_dropout_prob=0.3,
):
    config = RobertaConfig.from_json_file("output/config.json")
    config.attention_probs_dropout_prob = attention_probs_dropout_prob
    config.do_sample = True
    config.num_beams = 500
    config.hidden_dropout_prob = hidden_dropout_prob
    config.repetition_penalty = 5
    config.num_labels = 3

    config
    return batch_size, config, learning_rate, adam_epsilon


def hinglishRoberta(
    batch_size=8,
    attention_probs_dropout_prob=0.4,
    learning_rate=5e-7,
    adam_epsilon=1e-8,
    hidden_dropout_prob=0.3,
    input_name="Roberta",
):
    global name
    name = input_name
    open(f"{name}.log", "w").write(f"------ Starting for model {name}------\n")
    sentences, labels, le = load_sentences_and_labels()
    device_name = tf.test.gpu_device_name()
    device = check_for_gpu(device_name)
    tokenizer, input_ids = tokenize_the_sentences(sentences)
    input_ids, MAX_LEN = add_padding(tokenizer, input_ids)
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

    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def flat_prf(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return precision_recall_fscore_support(
            labels_flat, pred_flat, labels=[0, 1, 2], average="macro"
        )

    def format_time(elapsed):
        """
        Takes a time in seconds and returns a string hh:mm:ss
        """

        elapsed_rounded = int(round((elapsed)))

        return str(datetime.timedelta(seconds=elapsed_rounded))

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
        tokenizer, MAX_LEN, model, device, le, final_name="test.json"
    )
    full_output = evaluate_final_text(
        tokenizer, MAX_LEN, model, device, le, final_name="final_test.json"
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
    full_output.to_csv("Robertapreds.csv")

    output_dir = f"./{name}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    open(f"{name}.log", "a").write("Saving model to %s\n" % output_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def evaluate_final_text(tokenizer, MAX_LEN, model, device, le, final_name):
    final_test_df = pd.read_json(final_name)
    sentences = final_test_df["clean_text"]

    input_ids = []

    for sent in sentences:

        encoded_sent = tokenizer.encode(
            sent,
            add_special_tokens=True,
        )

        input_ids.append(encoded_sent)

    input_ids = pad_sequences(
        input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post"
    )

    attention_masks = []

    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)

    batch_size = 32

    prediction_data = TensorDataset(prediction_inputs, prediction_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=batch_size
    )

    open(f"{name}.log", "a").write(
        "Predicting labels for {:,} valid sentences...\n".format(len(prediction_inputs))
    )

    model.eval()

    predictions = []

    for batch in prediction_dataloader:

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask = batch

        with torch.no_grad():

            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()

        predictions.append(logits)

    open(f"{name}.log", "a").write("    DONE.\n")

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    proba = [item for sublist in predictions for item in sublist]
    preds = np.argmax(proba, axis=1).flatten()

    output = le.inverse_transform(flat_predictions.tolist())
    output_df = pd.DataFrame(
        {
            "Uid": list(final_test_df["uid"]),
            "Sentiment": output,
            "clean_text": list(final_test_df["clean_text"]),
        }
    )
    output_df.to_csv(f"{name}-{final_name[:-5]}-output-df.csv")
    proba = [item for sublist in predictions for item in sublist]
    preds = np.argmax(proba, axis=1).flatten()
    full_output = output_df
    full_output["proba_negative"] = pd.DataFrame(proba)[0]
    full_output["proba_neutral"] = pd.DataFrame(proba)[1]
    full_output["proba_positive"] = pd.DataFrame(proba)[2]
    full_output.to_csv(f"{name}-{final_name[:-5]}-full-output.csv")

    return full_output


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

            #

            #

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


def set_seed():
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def load_lm_model(config):
    model = RobertaForSequenceClassification.from_pretrained("output", config=config)
    model.cuda()
    params = list(model.named_parameters())
    return model


def make_dataloaders(
    train_inputs,
    train_masks,
    train_labels,
    batch_size,
    validation_inputs,
    validation_masks,
    validation_labels,
):

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size
    )

    validation_data = TensorDataset(
        validation_inputs, validation_masks, validation_labels
    )
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(
        validation_data, sampler=validation_sampler, batch_size=batch_size
    )
    return train_dataloader, validation_dataloader


def load_masks_and_inputs(input_ids, labels, attention_masks):

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        input_ids, labels, random_state=2018, test_size=0.1
    )

    train_masks, validation_masks, _, _ = train_test_split(
        attention_masks, labels, random_state=2018, test_size=0.1
    )

    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    return (
        train_inputs,
        train_masks,
        train_labels,
        validation_inputs,
        validation_masks,
        validation_labels,
    )


def create_attention_masks(input_ids):

    attention_masks = []

    for sent in input_ids:

        att_mask = [int(token_id > 0) for token_id in sent]

        attention_masks.append(att_mask)
    return attention_masks


def add_padding(tokenizer, input_ids):

    MAX_LEN = 300

    open(f"{name}.log", "a").write(
        "\nPadding/truncating all sentences to %d values...\n" % MAX_LEN
    )

    open(f"{name}.log", "a").write(
        '\nPadding token: "{:}", ID: {:}\n'.format(
            tokenizer.pad_token, tokenizer.pad_token_id
        )
    )

    input_ids = pad_sequences(
        input_ids,
        maxlen=MAX_LEN,
        dtype="long",
        value=0,
        truncating="post",
        padding="post",
    )

    open(f"{name}.log", "a").write("\nDone.")
    return input_ids, MAX_LEN


def tokenize_the_sentences(sentences):

    open(f"{name}.log", "a").write("Loading Roberta tokenizer...\n")
    tokenizer = RobertaTokenizer.from_pretrained("output")
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


def check_for_gpu(device_name):

    if device_name == "/device:GPU:0":
        open(f"{name}.log", "a").write("Found GPU at: {}\n".format(device_name))
        device = torch.device("cuda")
    else:
        raise SystemError("GPU device not found")
    return device


def load_sentences_and_labels():
    train_df = pd.read_json("train.json")
    test_df = pd.read_json("test.json")
    sentences = train_df["clean_text"]
    labels = train_df["sentiment"]
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    return sentences, labels, le
