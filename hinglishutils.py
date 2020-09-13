import datetime
import os
import random
import re
import time

import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    RobertaConfig,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix,
        index=class_names,
        columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=fontsize
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def get_files_from_gdrive(url: str, fname: str) -> None:
    file_id = url.split("/")[5]
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, fname, quiet=False)


def clean(df, col):
    """Cleaning Twiitter data

    Arguments:
        df {[pandas dataframe]} -- Dataset that needs to be cleaned
        col {[string]} -- column in which text is present

    Returns:
        [pandas dataframe] -- Datframe with a "clean_text" column
    """
    df["clean_text"] = df[col]
    df["clean_text"] = (
        (df["clean_text"])
        .apply(lambda text: re.sub(r"RT\s@\w+:", "Retweet", text))  # Removes RTS
        .apply(lambda text: re.sub(r"@", "mention ", text))  # Replaces @ with mention
        .apply(lambda text: re.sub(r"#", "hashtag ", text))  # Replaces # with hastag
        .apply(lambda text: re.sub(r"http\S+", "", text))  # Removes URL
    )
    return df


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


def all_the_important_configs(
    model,
    batch_size=8,
    attention_probs_dropout_prob=0.4,
    learning_rate=5e-7,
    adam_epsilon=1e-8,
    hidden_dropout_prob=0.3,
):
    if model == "bert":
        config = BertConfig.from_json_file("model_save/config.json")
    elif model == "distilbert":
        config = DistilBertConfig.from_json_file("distilBert6/config.json")
    elif model == "roberta":
        config = RobertaConfig.from_json_file("roberta3/config.json")
    config.attention_probs_dropout_prob = attention_probs_dropout_prob
    config.do_sample = True
    config.num_beams = 500
    config.hidden_dropout_prob = hidden_dropout_prob
    config.repetition_penalty = 5
    config.num_labels = 3

    config
    return batch_size, config, learning_rate, adam_epsilon


def check_for_gpu(device_name, name):

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


def evaluate_final_text(tokenizer, MAX_LEN, model, device, le, final_name, name):
    final_test_df = pd.read_json(final_name)
    sentences = final_test_df["clean_text"]

    prediction_inputs, prediction_masks = prep_input(sentences, tokenizer, MAX_LEN)

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

    predictions = get_preds_from_model(prediction_dataloader, device, model)

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


def get_preds_from_model(prediction_dataloader, device, model):
    predictions = []

    for batch in prediction_dataloader:

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask = batch

        with torch.no_grad():

            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()

        predictions.append(logits)
    return predictions


def prep_input(sentences, tokenizer, MAX_LEN):
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
    return prediction_inputs, prediction_masks


def set_seed():
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


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


def add_padding(tokenizer, input_ids, name):

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
