# -*- coding: utf-8 -*-
import time
import torch
from torch import nn
import numpy as np
import logging
from utils import dataset, metrics, config
import random
from tqdm import tqdm
# from visdom import Visdom
# from model.STCKA import *
from model.CRFA import *
from model.CRFA_NA import *
from model.CRFA_Stage1 import *
from model.CRFA_withoutContext import *
from model.CRFA_NC import *

import os

# viz = Visdom(env="CRFA")
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(model, train_iter, dev_iter, epoch, lr, loss_func):
    # total = sum(p.numel() for p in model.parameters())
    start_time = time.time()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.00001)

    all_loss = 0.0
    model.train()
    ind = 0.0
    for idx, batch in enumerate(train_iter):
        txt_text = batch.text[0]
        cpt_text = batch.concept[0]
        target = batch.label

        if torch.cuda.is_available():
            txt_text = txt_text.cuda()
            cpt_text = cpt_text.cuda()
            target = target.cuda()

        optim.zero_grad()

        logit = model(txt_text, cpt_text)
        loss = loss_func(logit, target)
        loss.backward()

        # clip_gradient(model, 1e-1)
        optim.step()

        if idx % 10 == 0:
            logger.info('Epoch:%d, Idx:%d, Training Loss:%.4f', epoch, idx, loss.item())

        all_loss += loss.item()

        ind += 1
    end_time = time.time()
    print("Train time:", end_time-start_time)

    eval_loss, acc, p, r, f1 = eval_model(model, dev_iter, loss_func)

    return all_loss / ind, eval_loss, acc, p, r, f1


def eval_model(model, val_iter, loss_func):
    start_time = time.time()
    eval_loss = 0.0
    ind = 0.0
    score = 0.0
    pred_label = None
    target_label = None
    # flag = True
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_iter)):
            txt_text = batch.text[0]
            cpt_text = batch.concept[0]
            target = batch.label
            if torch.cuda.is_available():
                txt_text = txt_text.cuda()
                cpt_text = cpt_text.cuda()
                target = target.cuda()
            logit = model(txt_text, cpt_text)

            loss = loss_func(logit, target)

            eval_loss += loss.item()
            if ind > 0:
                pred_label = torch.cat((pred_label, logit), 0)
                target_label = torch.cat((target_label, target))
            else:
                pred_label = logit
                target_label = target

            ind += 1

    acc, p, r, f1 = metrics.assess(pred_label, target_label)

    end_time = time.time()
    print("Eval time:", end_time - start_time)

    return eval_loss / ind, acc, p, r, f1


def main():
    start_time = time.time()
    #
    # seed = 1
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    PATH = "log/"

    args = config.config()
    lr = args.lr

    if not args.train_data_path:
        logger.info("please input train dataset path")
        exit()

    all_ = dataset.load_dataset(args.train_data_path, args.dev_data_path, args.test_data_path,args.txt_embedding_path, args.cpt_embedding_path, args.train_batch_size,args.dev_batch_size, args.test_batch_size)
    txt_TEXT, cpt_TEXT, txt_vocab_size, cpt_vocab_size, txt_word_embeddings, cpt_word_embeddings, train_iter, dev_iter, test_iter, label_size = all_
    print("train_data_path ",  args.train_data_path)
    print("label_size : ", label_size)
    model = CRFA(args.dropout, args.embedding_dim, label_size, txt_vocab_size, cpt_vocab_size, args.train_batch_size, args.hidden_size)
    # model = CRFA_withoutContext(args.dropout, args.embedding_dim, label_size, txt_vocab_size, cpt_vocab_size, args.train_batch_size, args.hidden_size)
    # model = CRFA_NC(args.dropout, args.embedding_dim, label_size, txt_vocab_size, cpt_vocab_size, args.train_batch_size, args.hidden_size)
    # model = CRFA_NA(args.dropout, args.embedding_dim, label_size, txt_vocab_size, cpt_vocab_size, args.train_batch_size, args.hidden_size)
    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    train_data, test_data = dataset.train_test_split(train_iter, 0.8)
    train_data, dev_data = dataset.train_dev_split(train_data, 0.8)
    loss_func = torch.nn.CrossEntropyLoss()

    if args.load_model:
        # model = torch.load(args.load_model)
        # model.load_state_dict(torch.load(args.load_model))
        test_loss, acc, p, r, f1 = eval_model(model, test_data, loss_func)
        logger.info('Test Loss:%.4f, Test Acc:%.4f, Test P:%.4f, Test R:%.4f, Test F1:%.4f', test_loss, acc, p, r, f1)
        return

    eval_acc, t_acc, trn_loss, dev_loss, trn_epoch = [], [], [], [], []
    best_score = 0.0
    test_best_score = 0.0
    max_test_p, max_test_r, max_test_f1 = 0.0, 0.0, 0.0
    test_loss, test_acc, test_p, test_r, test_f1 = 0, 0, 0, 0, 0
    for epoch in range(args.epoch):
        train_loss, eval_loss, acc, p, r, f1 = train_model(model, train_data, dev_data, epoch, args.lr, loss_func)
        logger.info('Epoch:%d, Training Loss:%.4f', epoch, train_loss)
        logger.info('Epoch:%d, Eval Loss:%.4f, Eval Acc:%.4f, Eval P:%.4f, Eval R:%.4f, Eval F1:%.4f', epoch, eval_loss,
                    acc, p, r, f1)

        # Reduce learning rate as number of epochs increase
        if (epoch % 2 == 0):
            lr = lr * (0.1 ** (epoch // 15))
            for param_group in optim.param_groups:
               param_group['lr'] = lr
               print("the set learning_rate is ", lr)

        if acc > best_score:
            best_score = acc
            # torch.save(model, 'model' + '_' + str(test_acc) + 'model.pth')
            torch.save(model.state_dict(), 'results/%d_%s_%s.pt' % (epoch, 'Model', str(best_score)))
            test_loss, test_acc, test_p, test_r, test_f1 = eval_model(model, test_data, loss_func)
            # save model--------------------------------------------------------
            # torch.save(model, 'model' + '_' + str(test_acc) + 'model.pth')
            if test_acc > test_best_score:
                test_best_score = test_acc

            if test_p > max_test_p:
                max_test_p = test_p
            if test_r > max_test_r:
                max_test_r = test_r
            if test_f1 > max_test_f1:
                max_test_f1 = test_f1

        logger.info('Test Loss:%.4f, Test Acc:%.4f, Test P:%.4f, Test R:%.4f, Test F1:%.4f', test_loss, test_acc, test_p, test_r, test_f1)

        # ACC，LOSS函数图像
        trn_epoch.append(epoch)
        trn_loss.append(train_loss)
        dev_loss.append(eval_loss)
        eval_acc.append(acc)
        t_acc.append(test_acc)
        # viz.line(X=np.column_stack((np.array(trn_epoch), np.array(trn_epoch))),
        #          Y=np.column_stack((np.array(trn_loss), np.array(dev_loss))),
        #          win='loss',
        #          opts=dict(legend=["train_Loss", "eval_loss"], title="loss"))
        #
        # viz.line(X=np.column_stack((np.array(trn_epoch), np.array(trn_epoch))),
        #          Y=np.column_stack((np.array(eval_acc), np.array(t_acc))),
        #          win='acc',
        #          opts=dict(legend=["eval_acc", "test_acc"], title="acc"))

    end_time = time.time()
    print("times:", end_time - start_time)
    print("test_Best_ACC:-------", test_best_score)
    print('max_test_p-------', max_test_p)
    print('max_test_r------', max_test_r)
    print('max_test_f1-----', max_test_f1)


if __name__ == "__main__":
    main()