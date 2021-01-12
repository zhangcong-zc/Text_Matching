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
        self.idf_path = './data/idf.npy'
        self.model_save_path = './save_model/'
        self.model_type = 'BCNN'    # BCNN ABCNN1 ABCNN2 ABCNN3
        self.vocab_size = 2000
        self.embedding_dim = 300
        self.seq_length = 20
        self.feature_size = 4
        self.learning_rate = 1e-5
        self.l2_reg = 0.0004
        self.keep_prob = 0.5
        self.hidden_dim = 256
        self.kernel_size = 3
        self.num_classes = 2
        self.num_layers = 2
        self.batch_size = 32
        self.epochs = 1000