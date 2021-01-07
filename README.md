# Writer-Style-Recognition
> 浙江大学《人工智能与系统》课程作业，作家风格识别
>
> 项目来源于：<https://mo.zju.edu.cn/workspace/5f8ed41148689dc2e7441a23?type=app&tab=2>（似乎只有我登录了自己的号才能看到。。。）

具体数据处理过程说明可查看 `torch_main.ipynb`



## 1 SVM

直接运行 `svm_train.py` 即可开始拟合。

之后运行 `nn_predict.py` 对预设的文字进行预测，输出结果。

通过 `model_test.py` 测试结果为 50 个中正确 45 个。



## 2 Pytorch 神经网络

直接运行 `nn_train.py` 即可开始训练

训练过程输出

```text
epoch:0 | valid_acc:0.2632
epoch:1 | valid_acc:0.4286
epoch:2 | valid_acc:0.6184
epoch:3 | valid_acc:0.8120
epoch:4 | valid_acc:0.8816
epoch:5 | valid_acc:0.9023
epoch:6 | valid_acc:0.8929
epoch:7 | valid_acc:0.9117
epoch:8 | valid_acc:0.9154
epoch:9 | valid_acc:0.9098
epoch:10 | valid_acc:0.8966
epoch:11 | valid_acc:0.9117
epoch:12 | valid_acc:0.9023
epoch:13 | valid_acc:0.9117
epoch:14 | valid_acc:0.9023
epoch:15 | valid_acc:0.9229
epoch:16 | valid_acc:0.8910
epoch:17 | valid_acc:0.9211
epoch:18 | valid_acc:0.9060
epoch:19 | valid_acc:0.9229
best accuracy:0.9436
```

之后运行 `nn_predict.py` 对预设的文字进行预测，输出结果

```text
LX
```

通过 `model_test.py` 测试结果为 50 个中正确 46 个。

## 3 bert

