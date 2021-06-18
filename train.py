import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from transformers import ElectraTokenizer, ElectraModel, ElectraForSequenceClassification, ElectraConfig
from transformers.integrations import WandbCallback
from transformers import XLMForSequenceClassification, XL
from load_data import *
import wandb

wandb.login()
os.environ['WANDB_PROJECT'] = 'wandb_test'
os.environ['WANDB_LOG_MODEL'] = 'true'

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def train():
  # load model and tokenizer
  # Bert
  #MODEL_NAME = "bert-base-multilingual-cased"
  # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # electra
  MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
  tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
  # pretrained 된 tokenizer model 불러오기 (bert base model.)

  # load dataset
  # train data, lobel token 불러오기
  train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
  ## load_data 를 통한 데이터 불러오기.
  #dev_dataset = load_data("./dataset/train/dev.tsv")
  train_label = train_dataset['label'].values
  #dev_label = dev_dataset['label'].values
  
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  #tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  #RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter
  # bert
  # bert_config = BertConfig.from_pretrained(MODEL_NAME)
  # bert_config.num_labels = 42
  # model = BertForSequenceClassification.from_pretrained(BertConfig)

  # electra
  electra_config = ElectraConfig.from_pretrained(MODEL_NAME)
  electra_config.num_labels = 42
  # electra_config.embedding_size = 512
  model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, config = electra_config)

  # WandbCallback().setup()

  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    seed = 1230,
    output_dir='./results',          # output directory
    save_total_limit=2,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=10,              # total number of training epochs
    learning_rate=5e-5,               # learning_rate
    per_device_train_batch_size=12,  # batch size per device during training
    #per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.03,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    #evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    report_to = 'wandb',
    run_name='maxlength_200_KoElectra_zero70%'
    #eval_steps = 500,            # evaluation step.
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    #eval_dataset=RE_dev_dataset,             # evaluation dataset
    #compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()

def main():
  train()

if __name__ == '__main__':
  main()
  """"
  훈련한다.
  """
