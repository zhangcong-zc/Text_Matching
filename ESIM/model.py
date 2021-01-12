# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/6/27 22:58
# @Author: Zhang Cong

import numpy as np
import tensorflow as tf
import tensorflow.contrib as contrib
from config import Config

class Model():
    def __init__(self):
        self.config = Config()                                                                                              # 读取配置参数
        self.input_query = tf.placeholder(shape=[None, self.config.seq_length], dtype=tf.int32, name="input-query")         # 输入query，One-Hot形式
        self.input_doc = tf.placeholder(shape=[None, self.config.seq_length], dtype=tf.int32, name="input-doc")             # 输入doc，One-Hot形式
        self.input_label = tf.placeholder(shape=[None, self.config.num_classes], dtype=tf.int32, name="input-label")        # 输入 label
        self.input_keep_prob = tf.placeholder(dtype=tf.float32, name='input-keep-prob')                                     # keep-prob

        # Embedding layer
        self.embedding = tf.get_variable(shape=[self.config.vocab_size, self.config.embedding_dim], dtype=tf.float32, name='embedding')

        # 将词汇映射为向量形式 [batch_size, seq_length, embedding_dim]
        self.input_query_emb = tf.nn.embedding_lookup(params=self.embedding, ids=self.input_query, name='input-query-emb')
        self.input_doc_emb = tf.nn.embedding_lookup(params=self.embedding, ids=self.input_doc, name='input-doc-emb')

        # 双向RNN编码 Input Encoding
        input_query_encode = self.bi_directional_rnn(input_data=self.input_query_emb, rnn_type=self.config.rnn_type, scope='Input_Encoding/Bi-LSTM')
        input_doc_encode = self.bi_directional_rnn(input_data=self.input_doc_emb, rnn_type=self.config.rnn_type, scope='Input_Encoding/Bi-LSTM', reuse=True)

        # query与doc局部交互层（Attention）
        with tf.name_scope('Local_inference_Modeling'):
            # 计算query与doc每个词语之间的相似度
            with tf.name_scope('word_similarity'):
                attention_weights = tf.matmul(input_query_encode, tf.transpose(input_doc_encode, [0, 2, 1]))
                attentionsoft_a = tf.nn.softmax(attention_weights)
                attentionsoft_b = tf.nn.softmax(tf.transpose(attention_weights, [0, 2, 1]))
                query_new = tf.matmul(attentionsoft_a, input_doc_encode)        # 使用doc向量生成new query向量
                doc_new = tf.matmul(attentionsoft_b, input_query_encode)        # 使用query向量生成new doc向量

            # 计算old_query与new_query的差、积
            query_diff = tf.subtract(input_query_encode, query_new)
            query_mul = tf.multiply(input_query_encode, query_new)

            # 计算old_doc与new_doc的差、积
            doc_diff = tf.subtract(input_doc_encode, doc_new)
            doc_mul = tf.multiply(input_doc_encode, doc_new)

            # 将原始query、new_query、差、积按维度进行特征拼接
            self.query_feature = tf.concat([input_query_encode, query_new, query_diff, query_mul], axis=2)
            self.doc_feature = tf.concat([input_doc_encode, doc_new, doc_diff, doc_mul], axis=2)

        with tf.name_scope("Inference_Composition"):
            # 双向RNN编码
            query_final = self.bi_directional_rnn(input_data=self.query_feature, rnn_type=self.config.rnn_type, scope='Inference_Composition/biLSTM')
            doc_final = self.bi_directional_rnn(input_data=self.doc_feature, rnn_type=self.config.rnn_type, scope='Inference_Composition/biLSTM', reuse=True)

            # 平均池化 average pool
            query_avg = tf.reduce_mean(query_final, axis=1)
            doc_avg = tf.reduce_mean(doc_final, axis=1)

            # 最大池化 max pool
            query_max = tf.reduce_max(query_final, axis=1)
            doc_max = tf.reduce_max(doc_final, axis=1)

            # 将四个池化特征进行维度拼接
            combine_emb = tf.concat([query_avg, query_max, doc_avg, doc_max], axis=1)

        # 全连接层 1
        with tf.variable_scope('feed_foward_layer1'):
            inputs = tf.nn.dropout(combine_emb, self.input_keep_prob)
            outputs = tf.layers.dense(inputs=inputs,
                                      units=self.config.hidden_dim,
                                      activation=tf.nn.relu,
                                      use_bias=True,
                                      kernel_initializer=tf.random_normal_initializer(0.0, 0.1))
        # 全连接层 2
        with tf.variable_scope('feed_foward_layer2'):
            outputs = tf.nn.dropout(outputs, self.input_keep_prob)
            self.logits = tf.layers.dense(inputs=outputs,
                                          units=self.config.num_classes,
                                          activation=tf.nn.tanh,
                                          use_bias=True,
                                          kernel_initializer=tf.random_normal_initializer(0.0, 0.1))
        # 类别score
        self.score = tf.nn.softmax(self.logits, name='score')
        # 预测结果
        self.predict = tf.argmax(self.score, axis=1, name='predict')
        # 准确率
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.input_label, axis=1), self.predict), dtype=tf.float32),name='accuracy')
        # 结构化损失函数，交叉熵+L2正则化
        self.loss = tf.add(
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_label)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            name="loss")
        # 优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate, name="optimizer").minimize(self.loss)


    def bi_directional_rnn(self, input_data, rnn_type, scope, reuse=False):
        '''
        构建双向RNN层，可选LSTM/GRU
        :param input_data: 输入时序数据
        :param rnn_type: RNN类型
        :param scope: 变量空间
        :param reuse: 是否重用变量
        :return:
        '''
        with tf.variable_scope(scope, reuse=reuse):
            cell_fw = self.get_rnn(rnn_type)
            cell_bw = self.get_rnn(rnn_type)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=input_data, dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
            return outputs


    def get_rnn(self, rnn_type):
        '''
        根据rnn_type创建RNN层
        :param rnn_type: RNN类型
        :return:
        '''
        if rnn_type == 'lstm':
            cell = contrib.rnn.LSTMCell(num_units=self.config.hidden_dim)
        else:
            cell = contrib.rnn.GRUCell(num_units=self.config.hidden_dim)
        cell = contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=self.input_keep_prob)
        return cell


if __name__ == '__main__':
    Model()