## Combining Context-relevant Features with Multi-stage Attention Network for Short Text Classification

#### CRFA模型代码的具体实现

### 目录结构
```
├─newmain.py
├─README.md
├─requirements.txt
├─utils
|   ├─config.py
|   ├─dataset.py
|   ├─metrics.py
├─results
├─model
|   ├─cnn.py
|   ├─CRFA.py
|   ├─CRFA_NA.py
|   ├─CRFA_NC.py
|   ├─CRFA_Stage1.py
|   ├─CRFA_withoutContext.py
|   ├─TCN.py
├─dataset
|    ├─preprocess.py
|    └TREC.tsv
```

### Requirements
* numpy==1.21.3
* requests==2.26.0
* scikit_learn==1.0.1
* spacy==3.2.3
* tagme==0.1.3
* torch==1.8.0+cu111
* torchtext==0.9.0
* tqdm==4.62.3

### 数据输入格式
demo示例为TREC数据集，
```
# 数据样例
# 原始文本 \t 相关概念 \t 标签
dist far denver aspen	city market place area community	num
city county modesto california	city county city community town area jurisdiction state area jurisdiction place large state	loc
desc galileo	desc galileo	hum
...
```

### 数据处理
相关文件下载地址，文件具体存放位置为dataset文件夹下
* [data-concept-instance-relations.txt](https://concept.research.microsoft.com/Home/Download)
* [glove](https://nlp.stanford.edu/projects/glove/)

运行 preprocess.py文件即可得到符合输入标准的数据集


### 运行方式
* 数据集划分
  * 原始数据集中80%作为训练集，20%作为测试集，训练集中的80%用于训练，20%用于验证
* 运行
```
$ python newmain.py --epoch 100 --lr 2e-4 --train_data_path dataset/TREC.tsv --txt_embedding_path dataset/glove.6B.300d.txt  --cpt_embedding_path dataset/glove.6B.300d.txt  --embedding_dim 300 --train_batch_size 128 --hidden_size 64

```
文件运行之前需要创建results文件夹来存储模型参数等信息，对于运行中可以更改的详细参数可见utils包下的config.py文件

### 代码文件说明
* dataset
  * preprocess.py : 预处理数据集，可以将数据集处理为上述所提到的输入数据格式
  * TREC.tsv : 运行样例
* model包
  * cnn.py : 一维卷积神经网络的实现
  * CRFA.py : CRFA模型
  * TCN.py : TCN模型
* util包
  * config.py　: 参数配置
  * dataset.py : 数据集的加载和处理，使数据集中的数据变为词向量
  * metrics.py : 评价指标函数的实现
* newmain.py : 运行文件

### 参考论文
```
Liu, Yingying, Peipei Li, and Xuegang Hu. Combining context-relevant features with multi-stage attention network for short text classification[J]. Computer Speech & Language, 2022, 71: 101268.
```