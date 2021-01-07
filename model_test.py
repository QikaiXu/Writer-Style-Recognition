import os

# from nn_predict import predict
from SVM_predict import predict
# from bert_predict import predict


test_data = []
path = 'test_data/test_case1_data'
for file in os.listdir(path):
    if not os.path.isdir(file) and not file[0] == '.':  # 跳过隐藏文件和文件夹
        with open(os.path.join(path, file), 'r',  encoding='UTF-8') as f:  # 打开文件
            for line in f.readlines():
                test_data.append((line, file[:-4]))


total = len(test_data)
count = 0
for text, label in test_data:
    label_pred = predict(text)
    if label == label_pred:
        count += 1

print('right / total: {} / {}'.format(count, total))
