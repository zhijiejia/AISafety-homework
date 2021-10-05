# Lseeon 2 TASK: A Classification network on Cifar10

目录
=================

   * [Introduction](#introduction)
   * [Installation](#installation)
   * [Running Single-Image Tasks](#running-single-image-tasks)
        * [Storing Representations](#storing-representations)
        * [Storing Predictions](#storing-predictions)
   * [Running Multi-Image Tasks](#running-multi-image-tasks)
   * [Training Data Statistics](#training-data-statistics)
   * [Citing](#citing)


<div align="center">
  <img src="assets/web_assets/task_dict_v.jpg" />
</div>

---

## **整体介绍**
1. 此仓库存放了人工智能安全课程第二次课程作业，搭建分类神经网络在Cifar10公开数据集上完成分类任务

2. 本次作业最终在Cifar10测试集上取得了94.3%的top-1分类准确度

---

## **文件结构**
    .
    ├── Cifar10_struct.png
    ├── __init__.py
    ├── net
    │   ├── CifarNet.py
    │   └── ResNet.py
    ├── readMe.md
    ├── test_parameter_grad_search.py
    ├── test.py
    └── tools
        ├── dataset.py
        ├── ext_transforms.py
        ├── losses.py
        ├── metric.py
        ├── scheduler.py
        ├── utils_bn.py
        └── utils.py

---
## **网络结构**

1. 本次作业的整体网络由 主干特征提取网络 + 特征分类网络 组成

2. 主干提取网络采用ResNet50, 并修改最后的全局池化层, 将全局池化为 1 像素, 改为全局池化为 4 像素, 保留了更多的图像特征

3. 特征分类网络采用线性层, 完成特征分类映射

---
## **训练细节**
1. 训练框架: pytorch

1. 优化器: SGD

2. 学习率: 0.1

3. 权重衰减: 5e-4

4. BatchSize: 128

5. 训练轮数: 200 epoch

6. 损失函数: 交叉熵损失

6. 学习率更新方式: WarmUp + Poly

7. 数据增强方式: RandomCrop、RandomHorizontalFlip、RandomRotation 

8. 数据记录: 实验数据采用tensorboard记录并绘制相关曲线图

---
## **开始训练**
```
python test.py

tensorboard --logdir runs
```

---
## **实验过程**

### 1. 检察初始loss是否合理

    实验训练结果发现, 网络初始loss在2.3左右, 对于10分类问题来说, 交叉熵的理论初始损失值

### 2. 训练参数网格搜索
    
    1. 训练参数采用网络搜索方式, 针对学习率, 优化器, 权重衰减系数进行网格化搜索, 从中选取效果较好的参数组合, 每一组参数训练 10 epoch, 并记录 10 轮中最好 Acc 成绩

    2. 学习率参数的可选值为: 0.1, 0.01, 0.001

    3. 优化器的可选值为: SGD、Adam

    4. 权重衰减的可选值为: 1e-3, 5e-4, 1e-4
    
### 3. 各个参数组合实验结果

|Base_lr       |Optimizer      |Weight Decay   |Acc      |
|:------------:|:-------------:|:-------------:|:-------:|
|0.1           |SGD            |1e-4           |66.32    |
|0.1           |SGD            |5e-4           |69.40    |
|0.1           |SGD            |1e-3           |59.61    |
|0.1           |Adam           |1e-4           |32.70    |
|0.1           |Adam           |5e-4           |33.29    |
|0.1           |Adam           |1e-3           |23.58    |
|||||
|0.01          |SGD            |1e-4           |68.40    |
|**0.01**      |**SGD**        |**5e-4**       |**69.63**|
|0.01          |SGD            |1e-3           |69.08    |
|0.01          |Adam           |1e-4           |49.47    |
|0.01          |Adam           |5e-4           |50.37    |
|0.01          |Adam           |1e-3           |37.58    |
|||||
|0.001         |SGD            |1e-4           |59.66    |
|0.001         |SGD            |5e-4           |59.84    |
|0.001         |SGD            |1e-3           |58.26    |
|0.001         |Adam           |1e-4           |66.62    |
|0.001         |Adam           |5e-4           |67.11    |
|0.001         |Adam           |1e-3           |66.32    | 


### 4. 网络超参数选取结果

根据步骤3中的实验结果, 选取 base_lr: 0.01, 优化器: SGD, weight decay: 5e-4, 作为网络的最终超参数, 进行5次实验, 计算每次模型的最优平均值与方差

---
## **网络最终实验结果**

---
## Citing

If you find the code or the models useful, please cite this paper:
```
@article{DBLP:journals/corr/HeZRS15,
  author    = {Kaiming He and
               Xiangyu Zhang and
               Shaoqing Ren and
               Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  journal   = {CoRR},
  volume    = {abs/1512.03385},
  year      = {2015},
  url       = {http://arxiv.org/abs/1512.03385},
  eprinttype = {arXiv},
  eprint    = {1512.03385},
  timestamp = {Wed, 17 Apr 2019 17:23:45 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/HeZRS15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


### License

The code and models are released under the MIT License (refer to the [LICENSE](https://github.com/StanfordVL/taskonomy/blob/master/LICENSE) file for details).


