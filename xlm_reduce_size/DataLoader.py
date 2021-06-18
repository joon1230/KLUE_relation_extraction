import pickle as pickle
import os
import pandas as pd
import torch


# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
    # label = []
    # for i in dataset[8]:
    #     if i == 'blind':
    #         label.append(100)
    #     else:
    #         label.append(label_type[i])

    label = []
    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            if label_type[i] != 0:
                l = 1
            else:
                l = 0
            label.append(l)
    length = 50
    sentences = []
    for i in dataset.values:
        sentence = i[1]
        e1 = [i[3], i[4]]
        e2 = [i[6], i[7]]
        if e1[0] > e2[0]:
            first_start = e2
            second_start = e1
        else:
            first_start = e1
            second_start = e2
        tmp = sorted(e1 + e2)
        if abs(tmp[2] - tmp[1]) < length * 2:
            s = sentence[max(first_start[0] - length, 0):second_start[1] + length]
        else:
            s = sentence[max(first_start[0] - length, 0):first_start[1] + length] + \
                sentence[max(second_start[0] - length, 0):second_start[1] + length]
        sentences.append(s)

    out_dataset = pd.DataFrame(
        {'sentence': sentences, 'entity_01': dataset[2], 'entity_02': dataset[5], 'label': label, })
    # df1 = out_dataset.loc[out_dataset.label == 0].sample(frac=0.7)
    # df2 = out_dataset.loc[out_dataset.label != 0]
    # out_dataset = pd.concat([df1, df2], axis=0)
    return out_dataset


# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
    # load label_type, classes
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    # preprecessing dataset
    dataset = preprocessing_dataset(dataset, label_type)

    return dataset


# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer):
    concat_entity = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = ''
        temp = e01 + "</s>" + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=150,
        add_special_tokens=True,
    )
    return tokenized_sentences
