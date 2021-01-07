import os
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

model_path = './results/'

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)


def load_data(path):
    """
    读取数据和标签
    :param path:数据集文件夹路径
    :return:返回读取的片段和对应的标签
    """
    sentences = []  # 片段
    target = []  # 作者

    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file) and not file[0] == '.':
            with open(os.path.join(path, file), 'r',  encoding='UTF-8') as f:  # 打开文件
                for line in f.readlines():
                    sentences.append(line)
                    target.append(labels[file[:-4]])
    return sentences, target


# Function to get token ids for a list of texts
def encode_fn(text_list):
    all_input_ids = []
    for text in text_list:
        input_ids = tokenizer.encode(
            text,
            add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
            max_length=160,  # 设定最大文本长度
            padding='max_length',  # pad到最大的长度
            return_tensors='pt',  # 返回的类型为pytorch tensor
            truncation=True
        )
        all_input_ids.append(input_ids)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids


def flat_accuracy(preds, labels):
    """A function for calculating accuracy scores"""

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


if __name__ == '__main__':
    if os.path.exists(model_path):
        print(model_path + " is exist")
    else:
        print(model_path + " is not exist")
        os.mkdir(model_path)
    x_train, y_train = load_data('./dataset')
    all_input_ids = encode_fn(x_train)
    labels = torch.tensor(y_train)
    epochs = 4
    batch_size = 16
    num_labels = 5
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Split data into train and validation
    dataset = TensorDataset(all_input_ids, labels)
    train_size = int(0.90 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # Load the pretrained BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=num_labels,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    model.to(device)

    # create optimizer and learning rate schedule
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        # 训练模型
        model.train()
        total_loss, total_val_loss = 0, 0
        total_eval_accuracy = 0
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
            outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                            labels=batch[1].to(device))
            loss = outputs[0]
            logits = outputs[1]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if step % 50 == 0:
                print("step: {0} loss: {1}".format(step, loss))
        # 验证模型
        model.eval()
        for i, batch in enumerate(val_dataloader):
            with torch.no_grad():
                outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                                labels=batch[1].to(device))
                loss = outputs[0]
                logits = outputs[1]
                total_val_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = batch[1].cpu().numpy()
                total_eval_accuracy += flat_accuracy(logits, label_ids)
                print("eval step: {0} loss: {1}".format(i, loss))
        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)

        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print("Saved model")
        print('Train loss     : {0}'.format(avg_train_loss))
        print('Validation loss: {0}'.format(avg_val_loss))
        print('Accuracy: {0}'.format(avg_val_accuracy))
