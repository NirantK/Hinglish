"""# Getting Training, Testing and Dev files for LM"""

import datetime
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from IPython.display import clear_output

# We'll borrow the `pad_sequences` utility function to do this.
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# Use train_test_split to split our data into train and validation sets for
# training
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
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
    config = BertConfig.from_json_file("model_save/config.json")
    config.attention_probs_dropout_prob = attention_probs_dropout_prob
    config.do_sample = True
    config.num_beams = 500
    config.hidden_dropout_prob = hidden_dropout_prob
    config.repetition_penalty = 5
    config.num_labels = 3

    config
    return batch_size, config, learning_rate, adam_epsilon


def hinglishbert(
    batch_size=8,
    attention_probs_dropout_prob=0.4,
    learning_rate=5e-7,
    adam_epsilon=1e-8,
    hidden_dropout_prob=0.3,
    input_name="bert",
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
        lr=learning_rate,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=adam_epsilon,  # args.adam_epsilon  - default is 1e-8.
    )
    epochs = 3
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,  # Default value in run_glue.py
        num_training_steps=total_steps,
    )

    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def flat_prf(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return precision_recall_fscore_support(labels_flat, pred_flat, labels=[0, 1, 2], average="macro")

    def format_time(elapsed):
        """
        Takes a time in seconds and returns a string hh:mm:ss
        """
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def run_valid():
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        open(f"{name}.log", "a").write("\n")
        open(f"{name}.log", "a").write("Running Validation...\n")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        eval_p = 0
        eval_r = 0
        eval_f1 = 0

        # Evaluate data for one epoch
        (
            eval_accuracy,
            nb_eval_steps,
            eval_p,
            eval_r,
            eval_f1,
        ) = evaluate_data_for_one_epochs(eval_accuracy, eval_p, eval_r, eval_f1, nb_eval_steps)
        open(f"{name}.log", "a").write("  Accuracy: {0:.2f}\n".format(eval_accuracy / nb_eval_steps))
        open(f"{name}.log", "a").write(
            f"  Precision, Recall F1: {eval_p/nb_eval_steps}, {eval_r/nb_eval_steps}, {eval_f1/nb_eval_steps}\n"
        )
        open(f"{name}.log", "a").write("  Validation took: {:}\n".format(format_time(time.time() - t0)))

    def evaluate_data_for_one_epochs(eval_accuracy, eval_p, eval_r, eval_f1, nb_eval_steps):
        for batch in validation_dataloader:

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            temp_eval_f1 = flat_prf(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy
            eval_p += temp_eval_f1[0]
            eval_r += temp_eval_f1[1]
            eval_f1 += temp_eval_f1[2]

            # Track the number of batches
            nb_eval_steps += 1
        return eval_accuracy, nb_eval_steps, eval_p, eval_r, eval_f1

    # Report progress.

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    set_seed()

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
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
    _ = evaluate_final_text(tokenizer, MAX_LEN, model, device, le, final_name="test.json")
    full_output = evaluate_final_text(tokenizer, MAX_LEN, model, device, le, final_name="final_test.json")

    l = pd.read_csv("test_labels_hinglish.txt")
    precision_recall_fscore_support(full_output["Sentiment"], l["Sentiment"][:-1], average="macro")
    open(f"{name}.log", "a").write(str(accuracy_score(full_output["Sentiment"], l["Sentiment"][:-1])))

    save_model(full_output, model, tokenizer)


def save_model(full_output, model, tokenizer):
    full_output.to_csv("bertpreds.csv")

    output_dir = f"./{name}/"

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    open(f"{name}.log", "a").write("Saving model to %s\n" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def evaluate_final_text(tokenizer, MAX_LEN, model, device, le, final_name):
    final_test_df = pd.read_json(final_name)
    sentences = final_test_df["clean_text"]
    # Report the number of sentences.

    # Create sentence and label lists
    # sentences = df.sentence.values
    # labels = df.label.values

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        )

        input_ids.append(encoded_sent)

    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)

    # Set the batch size.
    batch_size = 32

    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Prediction on valid set

    open(f"{name}.log", "a").write("Predicting labels for {:,} valid sentences...\n".format(len(prediction_inputs)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions = []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Store predictions and true labels
        predictions.append(logits)

    open(f"{name}.log", "a").write("    DONE.\n")

    # Combine the predictions for each batch into a single list of 0s and 1s.
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

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        open(f"{name}.log", "a").write("\n")
        # open(f"{name}.log", "a").write('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        open(f"{name}.log", "a").write("Training...\n")

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            clear_output(wait=True)
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                open(f"{name}.log", "a").write("======== Epoch {:} / {:} ========\n".format(epoch_i + 1, epochs))
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                open(f"{name}.log", "a").write(
                    "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.\n".format(step, len(train_dataloader), elapsed)
                )

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            # if step % 40 == 0 and not step == 0:
            #     # Calculate elapsed time in minutes.
            #     elapsed = format_time(time.time() - t0)
            #     run_valid()
        elapsed = format_time(time.time() - t0)
        run_valid()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        open(f"{name}.log", "a").write("")
        open(f"{name}.log", "a").write("  Average training loss: {0:.2f}\n".format(avg_train_loss))
        open(f"{name}.log", "a").write("  Training epcoh took: {:}\n".format(format_time(time.time() - t0)))

    open(f"{name}.log", "a").write("\n")
    open(f"{name}.log", "a").write("Training complete!\n")


def set_seed():
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def load_lm_model(config):
    model = BertForSequenceClassification.from_pretrained("model_save", config=config)
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
    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    return train_dataloader, validation_dataloader


def load_masks_and_inputs(input_ids, labels, attention_masks):
    # Use 90% for training and 10% for validation.
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        input_ids, labels, random_state=2018, test_size=0.1
    )
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state=2018, test_size=0.1)

    # Convert all inputs and labels into torch tensors, the required datatype
    # for our model.
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
    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    return attention_masks


def add_padding(tokenizer, input_ids):
    # Set the maximum sequence length.
    # I've chosen 64 somewhat arbitrarily. It's slightly larger than the
    # maximum training sentence length of 47...
    MAX_LEN = 300

    open(f"{name}.log", "a").write("\nPadding/truncating all sentences to %d values...\n" % MAX_LEN)

    open(f"{name}.log", "a").write(
        '\nPadding token: "{:}", ID: {:}\n'.format(tokenizer.pad_token, tokenizer.pad_token_id)
    )

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
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
    # Load the BERT tokenizer.
    open(f"{name}.log", "a").write("Loading BERT tokenizer...\n")
    tokenizer = BertTokenizer.from_pretrained("model_save")
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    open(f"{name}.log", "a").write("Tokenize the first sentence:\n")
    open(f"{name}.log", "a").write(str(tokenized_texts[0]))
    open(f"{name}.log", "a").write("\n")
    input_ids = []
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    # open(f"{name}.log", "a").write sentence 0, now as a list of IDs.

    return tokenizer, input_ids


def check_for_gpu(device_name):
    # The device name should look like the following:
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

    # !tar cvf hinglishBertClass.tar hinglishBertClass/

    # !cp hinglishBertClass.tar ../drive/My\ Drive/HinglishNLP/Models/Class
    # !cp bertpreds.csv ../drive/My\ Drive/HinglishNLP/PredOutput/
