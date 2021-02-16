"""
1. 总共 5 个作者，每个作者的前 200 个最高词频的词作为特征，共 1000 维（或者小于 1000 维）
2. 单独计算每个作者每一句话的这 1000 维特征
3. 用这个特征训练 SVM
结果：45 / 50
"""


import os
import jieba as jb
import jieba.analyse
import numpy as np
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


int2author = ['LX', 'MY', 'QZS', 'WXB', 'ZAL']
author_num = len(int2author)
author2int = {author: i for i, author in enumerate(int2author)}


# dataset = {(sentence, label), }
dataset_init = []
path = 'dataset/'
for file in os.listdir(path):
    if not os.path.isdir(file) and not file[0] == '.':  # 跳过隐藏文件和文件夹
        with open(os.path.join(path, file), 'r',  encoding='UTF-8') as f:  # 打开文件
            for line in f.readlines():
                dataset_init.append((line, author2int[file[:-4]]))


# 将片段组合在一起后进行词频统计
str_full = ['' for _ in range(author_num)]
for sentence, label in dataset_init:
    str_full[label] += sentence

# 词频特征统计，取出各个作家前 200 的词
words = set()
for label, text in enumerate(str_full):
    for word in jb.analyse.extract_tags(text, topK=200, withWeight=False):
        words.add(word)

int2word = list(words)
word_num = len(int2word)
word2int = {word: i for i, word in enumerate(int2word)}

features = np.zeros((len(dataset_init), word_num))
labels = np.zeros(len(dataset_init))
for i, (sentence, author_idx) in enumerate(dataset_init):
    feature = np.zeros(word_num, dtype=np.float)
    for word in jb.lcut(sentence):
        if word in words:
            feature[word2int[word]] += 1
    if feature.sum():
        feature /= feature.sum()
        features[i] = feature
        labels[i] = author_idx
    else:
        labels[i] = 5  # 表示识别不了作者

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)


# 训练一个 5 分类 svm
# print('training svm for all')
# start = time.time()
# svm = SVC(probability=True)
# svm.fit(X_train, y_train)
# print('Fitting time: {:.2f} s'.format(time.time() - start))
# print(svm.score(X_test, y_test))
# with open('results/svm_model.pkl', 'wb') as f:
#     pickle.dump((int2author, word2int, svm), f)
# print('saved model!')


# 为每个作家训练一个 svm，效果更好
start = time.time()
svm_lst = []
for i in range(author_num):
    svm_i = SVC(probability=True)
    y_train_i = [1 if j == i else 0 for j in y_train]
    y_test_i = [1 if j == i else 0 for j in y_test]
    print('training svm for', int2author[i])
    svm_i.fit(X_train, y_train_i)
    print('score:', svm_i.score(X_test, y_test_i))
    svm_lst.append(svm_i)

    end = time.time()
    print('Fitting time: {:.2f} s'.format(end - start))
    start = end

with open('results/svm_model.pkl', 'wb') as f:
    pickle.dump((int2author, word2int, svm_lst), f)
print('saved model!')

