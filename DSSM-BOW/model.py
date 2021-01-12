# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/5/12 22:58 
# @Author: Zhang Cong

from config import Config
import tensorflow as tf

class Model():
    def __init__(self):
        self.config = Config()                                                                                                                      # 配置参数
        self.input_query = tf.placeholder(shape=[None, self.config.vocab_size], dtype=tf.float32, name='input-query')                               # 输入query，BOW形式
        self.input_pos_doc = tf.placeholder(shape=[None, self.config.vocab_size], dtype=tf.float32, name='input-pos')                               # 输入pos_doc，BOW形式
        self.input_neg_doc = tf.placeholder(shape=[None, self.config.neg_doc_num, self.config.vocab_size], dtype=tf.float32, name='input-neg')      # 输入多个neg_doc，BOW形式
        self.input_keep_prob = tf.placeholder(dtype=tf.float32, name='input-keep-prob')                                                             # keep-prob

        # 全连接层layer1 (batch_size, 2000) -> (batch_size, 300)
        L1_N = 300
        l1_range = tf.sqrt(6/(self.config.vocab_size + L1_N))       # 原论文weight、bias范围初始化方式
        weight_1 = tf.get_variable(initializer=tf.random_uniform(shape=[self.config.vocab_size, L1_N], minval=-l1_range, maxval=l1_range), name='weight-1')
        bias_1 = tf.get_variable(initializer=tf.random_uniform(shape=[L1_N], minval=-l1_range, maxval=l1_range), name='bias-1')
        # 全连接
        query_l1 = tf.matmul(self.input_query, weight_1) + bias_1
        pos_doc_l1 = tf.matmul(self.input_pos_doc, weight_1, ) + bias_1
        neg_doc_l1 = tf.matmul(tf.reshape(self.input_neg_doc, shape=[-1, self.config.vocab_size]), weight_1, ) + bias_1
        # 激活函数 activation function
        query_l1 = tf.nn.tanh(query_l1)
        pos_doc_l1 = tf.nn.tanh(pos_doc_l1)
        neg_doc_l1 = tf.nn.tanh(neg_doc_l1)

        # 全连接层layer2 (batch_size, 300) -> (batch_size, 300)
        L2_N = 300
        l2_range = tf.sqrt(6/(L1_N + L2_N))     # 原论文weight、bias范围初始化方式
        weight_2 = tf.get_variable(initializer=tf.random_uniform(shape=[L1_N, L2_N], minval=-l2_range, maxval=l2_range), name='weight-2')
        bias_2 = tf.get_variable(initializer=tf.random_uniform(shape=[L2_N], minval=-l2_range, maxval=l2_range), name='bias-2')
        # 全连接
        query_l2 = tf.matmul(query_l1, weight_2) + bias_2
        pos_doc_l2 = tf.matmul(pos_doc_l1, weight_2) + bias_2
        neg_doc_l2 = tf.matmul(neg_doc_l1, weight_2) + bias_2
        # 激活函数 activation function
        query_l2 = tf.nn.tanh(query_l2)
        pos_doc_l2 = tf.nn.tanh(pos_doc_l2)
        neg_doc_l2 = tf.nn.tanh(neg_doc_l2)

        # 全连接层layer3 (batch_size, 300) -> (batch_size, 128)
        L3_N = 128
        l3_range = tf.sqrt(6/(L2_N + L3_N))     # 原论文weight、bias范围初始化方式
        weight_3 = tf.get_variable(initializer=tf.random_uniform(shape=[L2_N, L3_N], minval=-l3_range, maxval=l3_range), name='weight-3')
        bias_3 = tf.get_variable(initializer=tf.random_uniform(shape=[L3_N], minval=-l3_range, maxval=l3_range), name='bias-3')
        # 全连接
        query_l3 = tf.matmul(query_l2, weight_3) + bias_3
        pos_doc_l3 = tf.matmul(pos_doc_l2, weight_3) + bias_3
        neg_doc_l3 = tf.matmul(neg_doc_l2, weight_3) + bias_3
        # 激活函数 activation function
        query_l3_out = tf.tanh(query_l3)
        pos_doc_l3_out = tf.tanh(pos_doc_l3)
        neg_doc_l3_out = tf.tanh(neg_doc_l3)

        # 维度还原 [batch_size, neg_doc_num, hidden_dim]
        neg_doc_l3_out = tf.reshape(neg_doc_l3_out, shape=[-1, self.config.neg_doc_num, L3_N])

        # 计算query和pos_doc的Cosine
        query_dot_pos = tf.reduce_sum(tf.multiply(query_l3_out, pos_doc_l3_out), axis=1)            # query和pos_doc进行点乘
        query_l3_l2 = tf.sqrt(tf.reduce_sum(tf.square(query_l3_out), axis=1))                       # query的L2范数
        pos_doc_l3_l2 = tf.sqrt(tf.reduce_sum(tf.square(pos_doc_l3_out), axis=1))                   # pos_doc的L2范数
        self.query_pos_cosine = tf.expand_dims(query_dot_pos/(query_l3_l2*pos_doc_l3_l2), axis=1)   # 计算query和pos_doc的余弦值

        # 测试结果
        self.test_predict = tf.reshape(tensor=tf.round(self.query_pos_cosine), shape=[-1])
        # 批测试准确率
        self.accuracy_test = tf.reduce_mean(tf.round(self.query_pos_cosine))

        # 计算query和neg_doc的Cosine
        query_l3_out_flatten = tf.expand_dims(query_l3_out, axis=1)                                 # 扩充query矩阵维度
        query_dot_neg = tf.reduce_sum(tf.multiply(query_l3_out_flatten, neg_doc_l3_out), axis=2)    # query和neg_doc进行点乘
        neg_doc_l3_l2 = tf.sqrt(tf.reduce_sum(tf.square(neg_doc_l3_out), axis=2))                   # neg_doc的L2范数
        self.query_neg_cosine = query_dot_neg/(tf.expand_dims(query_l3_l2, axis=1)*neg_doc_l3_l2)   # 计算query和neg_doc的余弦值

        # 将pos_doc和neg_doc的cosine进行拼接为一个整体矩阵
        doc_cosine = tf.concat([self.query_pos_cosine, self.query_neg_cosine], axis=1)
        # score归一化
        doc_cosine_softmax = tf.nn.softmax(doc_cosine, axis=1)
        # 获取query与pos_doc的相似度
        prob = tf.slice(doc_cosine_softmax, begin=[0, 0], size=[-1, 1])
        # 训练结果
        self.train_predict = tf.reshape(tensor=tf.round(prob), shape=[-1])
        # 损失函数 负对数损失函数，提升pos_doc的score，抑制neg_doc的score
        self.loss = -tf.reduce_sum(tf.log(prob))
        # 优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(loss=self.loss)

        # 构造true label
        label = [[1]+[0]*self.config.neg_doc_num]               # true label: [1, 0, 0, 0, 0]
        labels = tf.tile(label, [self.config.batch_size, 1])    # 按batch_size的数量进行复制

        # 正确率
        correct = tf.equal(tf.argmax(doc_cosine_softmax, axis=1), tf.argmax(labels, axis=1))
        self.accuracy_train = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))



if __name__ == '__main__':
    Model()