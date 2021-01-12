# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/5/12 22:57 
# @Author: Zhang Cong

# 模型配置参数
class Config():
    def __init__(self):
        self.original_data_path = './data/train.txt'
        self.dev_data_path = './data/dev.txt'
        self.stopwords_path = './data/stopwords.txt'
        self.preprocess_path = './data/preprocessed_data.txt'
        self.vocab_path = './data/vocab.txt'
        self.label_path = './data/label.txt'
        self.model_save_path = './save_model/'
        self.vocab_size = 2000
        self.neg_doc_num = 4
        self.learning_rate = 1e-5
        self.keep_prob = 0.5
        self.batch_size = 32
        self.epochs = 100