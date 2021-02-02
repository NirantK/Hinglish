from hinglish import HinglishTrainer
import fire


def train(
    model_name,
    batch_size,
    attention_probs_dropout_prob,
    hidden_dropout_prob,
    learning_rate,
    adam_epsilon,
):
    hinglishbert = HinglishTrainer(
        model_name,
        batch_size=batch_size,
        attention_probs_dropout_prob=0.4,
        hidden_dropout_prob=0.3,
        learning_rate=5e-07,
        adam_epsilon=1e-08,
    )
    hinglishbert.train()
    hinglishbert.evaluate()


if __name__ == "__main__":
    fire.Fire(train)
