from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments, AutoTokenizer
from transformers import XLMRobertaForSequenceClassification, XLMRobertaConfig
from DataLoader import *

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predctions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy" : acc,
    }


def train():
    MODEL_NAME = "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


    train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
    train_label = train_dataset['label'].values

    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    electra_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
    electra_config.num_labels = 2
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config = electra_config)

    model.to(device)

    train_args = TrainingArguments(
        seed=1230,
        output_dir='/opt/ml/code/results',  # output directory
        save_total_limit=1,
        save_steps=2200,
        num_train_epochs=9,
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        warmup_steps=1000,
        weight_decay=0.03,
        logging_dir='./logs',
        logging_steps=100,
        report_to='wandb',
        run_name='XLMRoberta_Binary_ReduceSentenceQuery_Maxlength150'
    )

    trainer = Trainer(
        model = model,
        args = train_args,
        train_dataset = RE_train_dataset,
    )
    trainer.train()

def main():
    train()

if __name__ == '__main__':
    main()

