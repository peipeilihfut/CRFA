# coding=utf-8
import torch.nn as nn
import time
from torch.nn.utils import weight_norm
import torch
import numpy as np
from torch.autograd import Variable


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
          其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        """
                相当于一个Residual block

                :param n_inputs: int, 输入通道数，词嵌入的大小,embed_dim,
                :param n_outputs: int, 输出通道数，通道的个数,有多少个n_outputs就有多少个一维卷积，也就是卷积核的个数  n_kernel
                :param kernel_size: int, 卷积核尺寸 kernel_size
                :param stride: int, 步长，一般为1
                :param dilation: int, 膨胀系数
                :param padding: int, 填充系数
                :param dropout: float, dropout比率
        """
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()

        """
                TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
                对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
                对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。
        """
        layers = []
        num_levels = len(num_channels)
        print("num_levels: ", num_levels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]   # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)   # *作用是将输入迭代器拆成一个个元素

    def forward(self, x):
        '''
           x: size of(Batch, input_channel, seq_len)
            :return: size of(Batch, output_channel, seq_len)
        '''
        return self.network(x)
    #  self.network(x).shape 返回的结果(Batch, output_channel, seq_len)


class TCN(nn.Module):

    def __init__(self, embedding_dim, output_size, num_channels,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TCN, self).__init__()
        self.embedding_dim = embedding_dim

        self.output_size = output_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout

        self.tcn = TemporalConvNet(embedding_dim, num_channels, kernel_size, dropout=dropout)
        self.decoder = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        # self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):

        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.drop(input)
        # emb = self.drop(input)  # emb.shape:  torch.Size([64, 26, 300])
        # print("emb.transpose(1, 2): ", emb.transpose(1, 2).shape)   torch.Size([64, 300, 26])
        # y1 = self.tcn(emb.transpose(1, 2))  torch.Size([64, 100, 26])
        # print("y1: ", y1.shape)
        x = self.tcn(emb.transpose(1, 2))
        #print("x-----", x.shape)
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)  # 64,26,100
        y = self.decoder(y)    # 64,26,128
        return y.contiguous()
