from fastcore.utils import store_attr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AdamW
import pandas as pd
from transformers import get_linear_schedule_with_warmup
from hinglishutils import (
    add_padding,
    check_for_gpu,
    create_attention_masks,
    evaulate_and_save_prediction_results,
    load_lm_model,
    load_masks_and_inputs,
    load_sentences_and_labels,
    make_dataloaders,
    modify_transformer_config,
    save_model,
    set_seed,
    tokenize_the_sentences,
    train_model,
)
from datetime import datetime
import wandb


class HinglishTrainer:
    def __init__(
        self,
        model_path: str,
        batch_size: int = 8,
        attention_probs_dropout_prob: float = 0.4,
        learning_rate: float = 5e-7,
        adam_epsilon: float = 1e-8,
        hidden_dropout_prob: float = 0.3,
        epochs: int = 3,
        wname=None,
        drivepath="../drive/My\ Drive/HinglishNLP/repro",
    ):
        store_attr()
        self.timestamp = str(datetime.now().strftime("%d.%m.%y"))
        if not self.wname:
            self.wname = self.model_name
        self.config = modify_transformer_config(
            self.batch_size,
            self.attention_probs_dropout_prob,
            self.learning_rate,
            self.adam_epsilon,
            self.hidden_dropout_prob,
            self.model_path,
        )
        self.model_name = self.config.model_type
        wandb.init(
            project="hinglish",
            config={
                "batch_size": self.batch_size,
                "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
                "learning_rate": self.learning_rate,
                "adam_epsilon": self.adam_epsilon,
                "hidden_dropout_prob": self.hidden_dropout_prob,
                "epochs": self.epochs,
            },
            name=f"{self.wname} {self.timestamp}",
        )
        print({"Model Info": f"Setup self.model training for {model_name}"})
        self.device = check_for_gpu(self.model_name)

    def setup(self):
        sentences, labels, self.le = load_sentences_and_labels()
        self.tokenizer, input_ids = tokenize_the_sentences(
            sentences, self.model_name, self.model_path
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
        self.model = load_lm_model(self.config, self.model_name, self.model_path)
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
            self.device,
            optimizer,
            scheduler,
            loss_values,
            self.model_name,
            self.validation_dataloader,
        )

    def evaluate(
        self,
        dev_json="test.json",
        test_json="final_test.json",
        test_labels="test_labels_hinglish.txt",
    ):
        output = evaulate_and_save_prediction_results(
            self.tokenizer,
            self.MAX_LEN,
            self.model,
            self.device,
            self.le,
            final_name=dev_json,
            name=self.model_name,
        )

        full_output = evaulate_and_save_prediction_results(
            self.tokenizer,
            self.MAX_LEN,
            self.model,
            self.device,
            self.le,
            final_name=test_json,
            name=self.model_name,
        )
        l = pd.read_csv(test_labels)
        prf = precision_recall_fscore_support(
            full_output["Sentiment"], l["Sentiment"], average="macro"
        )
        wandb.log({"Precision": prf[0], "Recall": prf[1], "F1": prf[2]})
        wandb.log(
            {"Accuracy": str(accuracy_score(full_output["Sentiment"], l["Sentiment"]))}
        )
        save_model(full_output, self.model, self.tokenizer, self.model_name)
