import jieba as jb
import numpy as np
import pickle


# # 5 个分类器分类
with open('results/svm_model.pkl', 'rb') as f:
    int2author, word2int, svm_lst = pickle.load(f)


def predict(text):
    word_num = len(word2int)
    feature = np.zeros(word_num)
    for word in jb.lcut(text):
        if word in word2int:
            feature[word2int[word]] += 1
    feature /= feature.sum()
    probabilities = []
    for svm_i in svm_lst:
        pred_y = svm_i.predict_proba([feature])[0]
        probabilities.append(pred_y[1])

    author_idx = max(enumerate(probabilities), key=lambda x: x[1])[0]
    return int2author[author_idx]


# 1 个分类器分类
# with open('results/svm_model.pkl', 'rb') as f:
#     int2author, word2int, svm = pickle.load(f)
#
# def predict(text):
#     word_num = len(word2int)
#     feature = np.zeros(word_num)
#     for word in jb.lcut(text):
#         if word in word2int:
#             feature[word2int[word]] += 1
#     feature /= feature.sum()
#
#     author_idx = int(svm.predict([feature])[0])
#     return int2author[author_idx]


if __name__ == '__main__':
    target_text = "中国中流的家庭，教孩子大抵只有两种法。其一是任其跋扈，一点也不管，\
            骂人固可，打人亦无不可，在门内或门前是暴主，是霸王，但到外面便如失了网的蜘蛛一般，\
            立刻毫无能力。其二，是终日给以冷遇或呵斥，甚于打扑，使他畏葸退缩，彷佛一个奴才，\
            一个傀儡，然而父母却美其名曰“听话”，自以为是教育的成功，待到他们外面来，则如暂出樊笼的\
            小禽，他决不会飞鸣，也不会跳跃。"

    print(predict(target_text))
