import numpy as np
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification


model_path = './results/'


def predict(text):
    Tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    text_list = []
    labels = []
    text_list.append(text)
    label = 0
    labels.append(label)
    tokenizer = Tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'  # 返回的类型为 pytorch tensor
    )
    input_ids = tokenizer['input_ids']
    token_type_ids = tokenizer['token_type_ids']
    attention_mask = tokenizer['attention_mask']

    # model = model.cuda()
    model.eval()
    preds = []
    # for i, batch in enumerate(pred_dataloader):
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    preds += list(np.argmax(logits, axis=1))
    labels = {0: 'LX', 1: 'MY', 2: 'QZS', 3: 'WXB', 4: 'ZAL'}
    prediction = labels[preds[0]]
    return prediction


if __name__ == '__main__':
    target_text = "中国中流的家庭，教孩子大抵只有两种法。其一是任其跋扈，一点也不管，\
            骂人固可，打人亦无不可，在门内或门前是暴主，是霸王，但到外面便如失了网的蜘蛛一般，\
            立刻毫无能力。其二，是终日给以冷遇或呵斥，甚于打扑，使他畏葸退缩，彷佛一个奴才，\
            一个傀儡，然而父母却美其名曰“听话”，自以为是教育的成功，待到他们外面来，则如暂出樊笼的\
            小禽，他决不会飞鸣，也不会跳跃。"

    print(predict(target_text))
