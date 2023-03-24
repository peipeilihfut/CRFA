#coding=utf-8
import torch.nn.functional as F
from model.TCN import *
from model.cnn import Conv1d


# 不使用上下文信息进行对比实验
class CRFA_withoutContext(nn.Module):

    def __init__(self, dropout, embedding_dim, output_size, txt_vocab_size, concept_vocab_size, batch_size, hidden_size):

        super(CRFA_withoutContext, self).__init__()
        self.dropout = dropout
        self.embedding_dim = embedding_dim

        self.dropout = nn.Dropout(dropout)
        self.output_size = output_size

        self.word_embed = nn.Embedding(txt_vocab_size, embedding_dim)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = txt_vocab_size

        self.word_weight = nn.Parameter(torch.rand(embedding_dim, 1))
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
        self.fc = nn.Linear(1800, self.output_size)

    def word_attention(self, input_sentences):
        '''S
        在 每句话中，每个词由不同的重要性，我们赋予不同的权重，得到词的特征表示
        '''
        #  h为上下文的表示
        # print("input_sentenses:  ", input_sentences.shape)  torch.Size([64, 29, 300])
        # h = self.tcn(input_sentences)  # (64, 26, 128)
        u = torch.relu(torch.matmul(input_sentences, self.word_weight))  # 64, 30, 1
        # print("u--------", u.shape)
        # print("u", (torch.matmul(u, self.W_word_2).permute(0, 2, 1)/math.sqrt(self.hidden_size)))
        alpha1 = F.softmax(torch.matmul(u, self.W_word_2).permute(0, 2, 1), dim=2)
        # print("alpha1-------------", alpha1.shape)   # 64, 1, 30
        S = torch.bmm(alpha1, input_sentences)   # torch.Size([64, 1, 300])
        # print("S.shape--------------", S.shape)
        return S

    def cpt_attention(self, input_sentence):
        '''
        在 每句话中，每个词由不同的重要性，我们赋予不同的权重，得到词的特征表示
        '''
        #  h为上下文的表示
        # h = self.tcn(input_sentence)
        u = torch.relu(torch.matmul(input_sentence, self.cpt_weight))   # torch.Size([64, 26, 1])
        # print("u_cpt-------", u.shape)
        alpha1 = torch.softmax(torch.matmul(u, self.W_word_cpt).permute(0, 2, 1), dim=2)   # torch.Size([64, 1, 26])
        # print("alpha_cpt------", alpha1.shape)
        S = torch.bmm(alpha1, input_sentence)    # torch.Size([64, 1, 300])
        # print("s_cpt------", S.shape)
        return S

    def forward(self, x, cpt_word):

        # 1. word encoder-attention
        encoded_sents = self.word_embed(x)    # batch_size, sen_len, d_model 64,30,300
        # 1.1 word feature encoder
        # print("encoded_sents-----------", encoded_sents.shape)
        x = self.dropout(encoded_sents)
        res_begin = x
        x = self.word_attention(x)
        # print("x1------", x.shape)
        output_begin = x.squeeze(1)
        # print("output_begin-------", output_begin.shape)
        # 1.2 res_first stage
        x = self.con1(res_begin)[-1].permute(0, 2, 1)
        x = self.dropout(x)
        res_mid = x
        x = self.word_attention(x)
        output_mid = x.squeeze(1)
        # print("output_mid------", output_mid.shape)
        # 1.3 res_two stage
        x = self.con1(res_mid)[-1].permute(0, 2, 1)
        x = self.dropout(x)
        x = self.word_attention(x)
        output_third = x.squeeze(1)
        # print("output_third------", output_third.shape)
# ------------------------------------------------------------
#         '''
        # 2. cpt encoder-attention
        input_cpt = self.cpt_word_embed(cpt_word)  # [batch_sizes,17,embedding_dim ]
        # print('input_cpt--------', input_cpt)
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
        x = self.cpt_attention(x)
        cpt_output_third = x.squeeze(1)
        # '''


        A = torch.cat((output_begin, output_mid, output_third, cpt_output_begin, cpt_output_mid, cpt_output_third), -1)
        # A = torch.cat((output_begin, output_mid, output_third), -1)

        logits = self.fc(A)
        # print("logits--------", logits.shape)    # [64, 7]

        return logits
