import torch.nn.functional as F
from model.TCN import *
from model.cnn import Conv1d


class CRFA_NA(nn.Module):

    def __init__(self, dropout, embedding_dim, output_size, txt_vocab_size, concept_vocab_size, batch_size, hidden_size):

        super(CRFA_NA, self).__init__()
        self.dropout = dropout
        self.embedding_dim = embedding_dim

        self.dropout = nn.Dropout(dropout)
        self.output_size = output_size

        self.word_embed = nn.Embedding(txt_vocab_size, embedding_dim)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = txt_vocab_size

        self.word_weight = nn.Parameter(torch.rand(hidden_size, 1))
        self.cpt_weight = nn.Parameter(torch.rand(hidden_size, 1))

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
        # self.linear = nn.Linear(768, self.output_size)
        self.fc = nn.Linear(120, self.output_size)

    def word_attention(self, input_sentences):
        '''S
        在 每句话中，每个词由不同的重要性，我们赋予不同的权重，得到词的特征表示
        '''
        #  h为上下文的表示
        # print("input_sentenses:  ", input_sentences.shape)  torch.Size([64, 29, 256])
        h = self.tcn(input_sentences)  # (64, 13, 128)
        h = torch.matmul(h, self.word_weight)
        # print('h-----', h.shape)

        return h

    def cpt_attention(self, input_sentence):
        '''
        在 每句话中，每个词由不同的重要性，我们赋予不同的权重，得到词的特征表示
        '''
        #  h为上下文的表示
        h = self.tcn(input_sentence)
        h = torch.matmul(h, self.word_weight)
        # print(h.shape)

        return h

    def forward(self, x, cpt_word):

        # 1. word encoder-attention
        encoded_sents = self.word_embed(x)    # batch_size, sen_len, d_model
        # print('encoded_sents-------', encoded_sents.shape)   # torch.Size([64, 13, 300])
        # 1.1 word feature encoder
        x = self.dropout(encoded_sents)
        res_begin = x
        x = self.word_attention(x)
        # print('x---------', x.shape)   # torch.Size([64, 1, 128])
        output_begin = x.squeeze(2)
        # print("out_begin------", output_begin.shape)    # torch.Size([64, 128])

        # 1.2 res_first stage
        x = self.con1(res_begin)[-1].permute(0, 2, 1)
        x = self.dropout(x)
        res_mid = x
        x = self.word_attention(x)
        # print("x_output_mid-----", x.shape)    # torch.Size([64, 1, 128])
        output_mid = x.squeeze(2)

        # 1.3 res_two stage
        x = self.con1(res_mid)[-1].permute(0, 2, 1)
        x = self.dropout(x)
        x = self.word_attention(x)
        # print("x_output_third------", x.shape)
        output_third = x.squeeze(2)

# ------------------------------------------------------------
        # 2. cpt encoder-attention
        input_cpt = self.cpt_word_embed(cpt_word)  # [batch_sizes,17,embedding_dim ]
        # print('input_cpt--------', input_cpt)
        # cpt_attention = self.cpt_attention(input_cpt).transpose(0, 1).squeeze(0)
        x = self.dropout(input_cpt)
        cpt_res_begin = x
        x = self.cpt_attention(x)
        cpt_output_begin = x.squeeze(2)

        # 1.2 res_first stage
        x = self.con1(cpt_res_begin)[-1].permute(0, 2, 1)
        x = self.dropout(x)
        cpt_res_mid = x
        x = self.cpt_attention(x)
        cpt_output_mid = x.squeeze(2)

        # 1.3 res_two stage
        x = self.con1(cpt_res_mid)[-1].permute(0, 2, 1)
        x = self.dropout(x)
        x = self.cpt_attention(x)
        cpt_output_third = x.squeeze(2)
        # print("cpt_output_third---------", cpt_output_third.shape)
        A = torch.cat((output_begin, output_mid, output_third, cpt_output_begin, cpt_output_mid, cpt_output_third), -1)
        # print("A-----", A.shape)
        logits = self.fc(A)
        # print("logits--------", logits.shape)    # [64, 7]

        return logits
