
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TCN import TCN
from model.cnn import *


class CRFA(nn.Module):

    def __init__(self, dropout, embedding_dim, output_size, txt_vocab_size, concept_vocab_size, batch_size, hidden_size):

        super(CRFA, self).__init__()
        self.dropout = dropout
        self.embedding_dim = embedding_dim

        self.dropout = nn.Dropout(dropout)
        self.output_size = output_size

        self.word_embed = nn.Embedding(txt_vocab_size, embedding_dim)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = txt_vocab_size

        self.word_weight = nn.Parameter(torch.rand(hidden_size, 1))
        self.cpt_weight = nn.Parameter(torch.rand(embedding_dim, 1))

        self.W_word_2 = nn.Parameter(torch.rand(1, 1))
        self.W_word_cpt = nn.Parameter(torch.rand(1, 1))

        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size, bias=False)

        self.cpt_word_embed = nn.Embedding(concept_vocab_size, embedding_dim)

        self.tcn = TCN(embedding_dim=embedding_dim, output_size=hidden_size, num_channels=[100, 100, 100], kernel_size=2,
                        dropout=0.5, emb_dropout=0.25)

        self.con1 = Conv1d(embedding_dim, embedding_dim, [1, 1])
        # Fully-Connected Layer
        self.relu = nn.ReLU()
        self.fc = nn.Linear(768, self.output_size)

    def word_attention(self, input_sentences):
        '''S
        在 每句话中，每个词由不同的重要性，我们赋予不同的权重，得到词的特征表示
        '''
        #  h为上下文的表示
        h = self.tcn(input_sentences)  # (64, 13, 128)

        u = torch.relu(torch.matmul(h, self.word_weight))
        alpha1 = F.softmax(torch.matmul(u, self.W_word_2).permute(0, 2, 1), dim=2)
        S = torch.bmm(alpha1, h)   # torch.Size([64, 1, 128])
        return S

    def cpt_attention(self, input_sentence):
        '''
        在 每句话中，每个词由不同的重要性，我们赋予不同的权重，得到词的特征表示
        '''
        #  h为上下文的表示
        h = self.tcn(input_sentence)

        u = torch.relu(torch.matmul(input_sentence, self.cpt_weight))   # torch.Size([64, 26, 1])
        alpha1 = torch.softmax(torch.matmul(u, self.W_word_cpt).permute(0, 2, 1), dim=2)   # torch.Size([64, 1, 26])
        S = torch.bmm(alpha1, h)    # torch.Size([64, 1, 128])

        return S

    def forward(self, x, cpt_word):

        # 1. word encoder-attention
        encoded_sents = self.word_embed(x)    # batch_size, sen_len, d_model
        # 1.1 word feature encoder
        x = self.dropout(encoded_sents)
        res_begin = x
        x = self.word_attention(x)
        output_begin = x.squeeze(1)

        # 1.2 res_first stage
        x = self.con1(res_begin)[-1].permute(0, 2, 1)
        x = self.dropout(x)
        res_mid = x
        x = self.word_attention(x)
        output_mid = x.squeeze(1)
        #
        # 1.3 res_two stage
        x = self.con1(res_mid)[-1].permute(0, 2, 1)
        x = self.dropout(x)
        res_three = x
        x = self.word_attention(x)
        output_third = x.squeeze(1)

        # 第四阶段
        # x = self.con1(res_three)[-1].permute(0, 2, 1)
        # x = self.dropout(x)
        # x = self.word_attention(x)
        # # print("x_output_third------", x.shape)
        # output_fourth = x.squeeze(1)

# ------------------------------------------------------------
        # 2. cpt encoder-attention
        input_cpt = self.cpt_word_embed(cpt_word)  # [batch_sizes,17,embedding_dim ]
        # cpt_attention = self.cpt_attention(input_cpt).transpose(0, 1).squeeze(0)
        x = self.dropout(input_cpt)
        cpt_res_begin = x
        x = self.cpt_attention(x)
        cpt_output_begin = x.squeeze(1)

        # 1.2 res_first stage
        x = self.con1(cpt_res_begin)[-1].permute(0, 2, 1)
        x = self.dropout(x)
        cpt_res_mid = x
        x = self.cpt_attention(x)
        cpt_output_mid = x.squeeze(1)

        # 1.3 res_two stage
        x = self.con1(cpt_res_mid)[-1].permute(0, 2, 1)
        x = self.dropout(x)
        cpt_res_three = x
        x = self.cpt_attention(x)
        cpt_output_third = x.squeeze(1)

        # 第四阶段
        # x = self.con1(cpt_res_three)[-1].permute(0, 2, 1)
        # x = self.dropout(x)
        # x = self.word_attention(x)
        # # print("x_output_third------", x.shape)
        # cpt_output_fourth = x.squeeze(1)

        A = torch.cat((output_begin, output_mid, output_third, cpt_output_begin, cpt_output_mid, cpt_output_third), -1)
        logits = self.fc(A)

        return logits
