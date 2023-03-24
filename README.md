# Combining Context-relevant Features with Multi-stage Attention Network for Short Text Classification

#### For the purpose of reproducing this paper, we implemented this code.

## Requirements
* numpy==1.21.3
* requests==2.26.0
* scikit_learn==1.0.1
* spacy==3.2.3
* tagme==0.1.3
* torch==1.8.0+cu111
* torchtext==0.9.0
* tqdm==4.62.3

## 数据输入格式
demo示例为TREC数据集，

```
# origin text \t concepts \t label
dist far denver aspen	city market place area community	num
city county modesto california	city county city community town area jurisdiction state area jurisdiction place large state	loc
desc galileo	desc galileo	hum
...
```

## 数据处理
相关文件下载地址
* [data-concept-instance-relations.txt](https://concept.research.microsoft.com/Home/Download)
* [glove](https://nlp.stanford.edu/projects/glove/)

运行 preprocess.py文件即可得到符合输入标准的数据集

## How to run
Train & Dev & Test:
Original dataset is randomly split into 80% for training and 20% for test. 20% of randomly selected training instances are used to form development set.

```
$ python main.py --epoch 100 --lr 2e-4 --train_data_path dataset/tagmynews.tsv --txt_embedding_path dataset/glove.6B.300d.txt  --cpt_embedding_path dataset/glove.6B.300d.txt  --embedding_dim 300 --train_batch_size 128 --hidden_size 64
```

More detailed configurations can be found in `config.py`, which is in utils folder.

## Cite
```
Chen J, Hu Y, Liu J, et al. Deep Short Text Classification with Knowledge Powered Attention[J]. 2019.
```

## Disclaimer

The code is for research purpose only and released under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).