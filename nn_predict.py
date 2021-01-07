import torch
import torch.nn as nn
import jieba as jb


config_path = 'results/nn_model.pth'
config = torch.load(config_path)

word2int = config['word2int']
int2author = config['int2author']
word_num = len(word2int)
model = nn.Sequential(
    nn.Linear(word_num, 512),
    nn.ReLU(),
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 6),
)
model.load_state_dict(config['model'])
int2author.append(int2author[0])


def predict(text):
    feature = torch.zeros(word_num)
    for word in jb.lcut(text):
        if word in word2int:
            feature[word2int[word]] += 1
    feature /= feature.sum()
    model.eval()
    out = model(feature.unsqueeze(dim=0))
    pred = torch.argmax(out, 1)[0]
    return int2author[pred]


if __name__ == '__main__':
    target_text = "中国中流的家庭，教孩子大抵只有两种法。其一是任其跋扈，一点也不管，\
            骂人固可，打人亦无不可，在门内或门前是暴主，是霸王，但到外面便如失了网的蜘蛛一般，\
            立刻毫无能力。其二，是终日给以冷遇或呵斥，甚于打扑，使他畏葸退缩，彷佛一个奴才，\
            一个傀儡，然而父母却美其名曰“听话”，自以为是教育的成功，待到他们外面来，则如暂出樊笼的\
            小禽，他决不会飞鸣，也不会跳跃。"

    print(predict(target_text))
