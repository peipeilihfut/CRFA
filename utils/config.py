# -*- coding: utf-8 -*-
import logging
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def config():
    parser = argparse.ArgumentParser()
    # parameters
    parser.add_argument("--epoch", default=100, type=int,
                        help="the number of epoches needed to train")
    parser.add_argument("--lr", default=0.0003, type=float,
                        help="the learning rate")
    parser.add_argument("--train_data_path", default="dataset/TREC.tsv", type=str,
                        help="train dataset path")
    # parser.add_argument("--train_data_path", default="dataset/tagmynews.tsv", type=str,
    #                     help="train dataset path")
    parser.add_argument("--dev_data_path", default=None, type=str,
                        help="dev dataset path")
    parser.add_argument("--test_data_path", default=None, type=str,
                        help="test dataset path")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="the batch size")
    parser.add_argument("--dev_batch_size", default=64, type=int,
                        help="the batch size")
    parser.add_argument("--test_batch_size", default=64, type=int,
                        help="the batch size")
    parser.add_argument("--num_layers", default=1, type=int,
                        help="the batch size")
    parser.add_argument("--dropout", default=0.5, type=float,
                        help="the batch size")

    parser.add_argument("--txt_embedding_path", default="dataset/glove.6B.300d.txt",
                                            type=str,
                                            help="target pre-trained word embeddings path")
    parser.add_argument("--cpt_embedding_path", default="dataset/glove.6B.300d.txt",
                        type=str,
                        help="target pre-trained word embeddings path")
    parser.add_argument("--embedding_dim", default=300, type=int,
                        help="the text/concept word embedding size")
    parser.add_argument("--hidden_size", default=128, type=int,
                        help="the hidden size")
    parser.add_argument("--output_size", default=8, type=int,
                        help="the output size")  #
    parser.add_argument("--fine_tuning", default=True, type=bool,
                        help="whether fine-tune word embeddings")
    parser.add_argument("--early_stopping", default=15, type=int,
                        help="Tolerance for early stopping (# of epochs).")

    parser.add_argument("--load_model", default=None,
                        help="load pretrained model for testing")

    # parser.add_argument("--d_ff", default=512, type=int,
    #                     help=" ")
    # parser.add_argument("--h", default=8, type=int,
    #                     help=" the number of head.")
    # parser.add_argument("--N", default=1, type=int,
    #                     help="the layer number of transformer")
    # parser.add_argument("--d_model", default=256, type=int)


    '''
    N = 1 #6 in Transformer Paper
    d_model = 256 #512 in Transformer Paper
    d_ff = 512 #2048 in Transformer Paper
    h = 8
    dropout = 0.1
    output_size = 4
    lr = 0.0003
    max_epochs = 35
    batch_size = 128
    max_sen_len = 60
    '''

    #  ----------------------------CNN-----------------------------------

    # parser.add_argument("--num_channels", default=256, type=int)
    # parser.add_argument("--kernel_size", default=[3, 4, 5])
    # parser.add_argument("--max_sen_len", default=30, type=int)
    # parser.add_argument("--hidden_layers", default=1, type=int)
    # parser.add_argument("--seq_len", default=300, type=int)
    #
    # parser.add_argument("--hidden_size_linear", default=64, type=int)

    args = parser.parse_args()

    return args
