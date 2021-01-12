# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/6/27 22:58
# @Author: Zhang Cong

import numpy as np
import tensorflow as tf
from config import Config

class Model():
    def __init__(self):
        self.config = Config()                                                                                              # 配置参数
        self.input_query = tf.placeholder(shape=[None, self.config.seq_length], dtype=tf.int32, name="input-query")         # 输入query，One-Hot形式
        self.input_doc = tf.placeholder(shape=[None, self.config.seq_length], dtype=tf.int32, name="input-doc")             # 输入doc，One-Hot形式
        self.input_label = tf.placeholder(shape=[None, self.config.num_classes], dtype=tf.int32, name="input-label")        # 输入label
        self.input_feature = tf.placeholder(tf.float32, shape=[None, self.config.feature_size], name="input-features")      # 额外特征 [文本1长度，文本2长度，两个文本的字符交集，sum(IDF)]

        # Embedding layer
        self.embedding = tf.get_variable(shape=[self.config.vocab_size, self.config.embedding_dim], dtype=tf.float32, name='embedding')

        # 将词汇映射为向量形式 [batch_size, seq_length, embedding_dim]
        self.input_query_emb = tf.nn.embedding_lookup(params=self.embedding, ids=self.input_query, name='input-query-emb')
        self.input_doc_emb = tf.nn.embedding_lookup(params=self.embedding, ids=self.input_doc, name='input-doc-emb')

        # 维度扩充[batch_size, seq_length, embedding_dim, 1]
        self.input_query_emb = tf.expand_dims(input=self.input_query_emb, axis=-1)
        self.input_doc_emb = tf.expand_dims(input=self.input_doc_emb, axis=-1)

        # 对输入进行全局池化 all pool
        input_query_all_pool = self.all_pool(variable_scope="input-left", x=self.input_query_emb)
        input_doc_all_pool = self.all_pool(variable_scope="input-right", x=self.input_doc_emb)

        # 第一次pad + 宽卷积
        left_wp, left_ap, right_wp, right_ap = self.CNN_layer(variable_scope="CNN-1",
                                                              query=self.input_query_emb,
                                                              doc=self.input_doc_emb,
                                                              dim=self.config.embedding_dim)

        # 将每个conv stack 的结果取all-pool，然后计算query与doc的cosine值作为额外特征
        sims = [self.cos_sim(input_query_all_pool, input_doc_all_pool), self.cos_sim(left_ap, right_ap)]

        # 如果conv layer 有2层（原论文中最多2层）
        if self.config.num_layers > 1:
            left_wp, left_ap, right_wp, right_ap = self.CNN_layer(variable_scope="CNN-2",
                                                                  query=left_wp,
                                                                  doc=right_wp,
                                                                  dim=self.config.hidden_dim)
            # 将第2层产生的cos特征加入额外特征列表
            sims.append(self.cos_sim(left_ap, right_ap))

        with tf.variable_scope("output-layer"):
            # 将额外字符特征feature与sims层次相似度特征进行拼接 [batch_size, 7]
            self.output_features = tf.concat([self.input_feature, tf.stack(sims, axis=1)], axis=1, name="output_features")
            # 全连接层
            self.logits = tf.contrib.layers.fully_connected(
                inputs=self.output_features,
                num_outputs=self.config.num_classes,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.config.l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

            # 还可使用 layer.dense 进行全连接
            # tf.layers.dense(inputs=self.output_features,
            #                 units=self.config.num_classes,
            #                 activation=None,
            #                 kernel_initializer=contrib.layers.xavier_initializer(),
            #                 kernel_regularizer=contrib.layers.l2_regularizer(scale=self.config.l2_reg),
            #                 bias_initializer=tf.constant_initializer(1e-04),
            #                 name='FC')

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


    def pad_for_wide_conv(self, x):
        '''
        对input进行padding，为宽卷积做预处理，在文本的首尾都填充 kernel_size - 1 个0
        :param x: 输入向量[batch_size, seq_length, hidden_dim, channel)]
        :return:
        '''
        return tf.pad(tensor=x,
                      paddings=np.array([[0, 0], [self.config.kernel_size - 1, self.config.kernel_size - 1], [0, 0], [0, 0]]),
                      mode="CONSTANT",
                      name="pad_wide_conv")


    def cos_sim(self, v1, v2):
        '''
        计算cosin值
        :param v1: 输入向量1 [v1, v2, v3 ...]
        :param v2: 输入向量2 [v1, v2, v3 ...]
        :return:
        '''
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
        dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

        return dot_products / (norm1 * norm2)


    def euclidean_score(self, v1, v2):
        '''
        计算attention weight
        原始论文提出的计算attention的方法，在实际过程中反向传播计算梯度时 容易出现NaN的情况
        :param v1: 矩阵1 [batch_size, seq_length, hidden]
        :param v2: 矩阵2 [batch_size, seq_length, hidden]
        :return:
        '''
        euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1))
        return 1 / (1 + euclidean)


    def make_attention_mat(self, x1, x2):
        '''
        计算attention weight
        作者论文中提出计算attention的方法 在实际过程中反向传播计算梯度时 容易出现NaN的情况 这里面加以修改
        :param x1: 矩阵1 [batch_size, seq_length, hidden]
        :param x2: 矩阵2 [batch_size, seq_length, hidden]
        :return:
        '''
        x2 = tf.transpose(tf.squeeze(x2, axis=-1), [0, 2, 1])
        attention = tf.einsum("ijk,ikl->ijl", tf.squeeze(x1, axis=-1), x2)
        return attention


    def convolution(self, name_scope, x, dim, reuse):
        '''
        卷积层函数
        :param name_scope: 该操作所属变量空间
        :param x: 输入四维矩阵[batch_size, seq_length, hidden_dim, channel]
        :param dim: 卷积核宽度（词向量大小 or 隐藏层向量大小）
        :param reuse: 是否复用，与已存在的相同命名层共享参数
        :return:
        '''
        with tf.name_scope(name_scope + "-conv"):
            with tf.variable_scope("conv") as scope:
                conv = tf.contrib.layers.conv2d(
                    inputs=x,
                    num_outputs=self.config.hidden_dim,
                    kernel_size=(self.config.kernel_size, dim),
                    stride=1,
                    padding="VALID",
                    activation_fn=tf.nn.tanh,
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.config.l2_reg),
                    biases_initializer=tf.constant_initializer(1e-04),
                    reuse=reuse,
                    trainable=True,
                    scope=scope
                )
                conv = tf.transpose(conv, [0, 1, 3, 2], name="conv_trans")
                return conv


    def w_pool(self, variable_scope, x, attention):
        '''
        权重池化函数
        :param variable_scope: 命名空间
        :param x: 输入向量 [batch, s+w-1, dim, 1]
        :param attention: 注意力权重 [batch, s+w-1]
        :return:
        '''
        model_type = self.config.model_type
        with tf.variable_scope(variable_scope + "-w_pool"):
            if model_type == "ABCNN2" or model_type == "ABCNN3":
                pools = []
                # 维度扩充 [batch, s+w-1] => [batch, s+w-1, 1, 1]
                attention = tf.expand_dims(tf.expand_dims(attention, -1), -1)
                # 进行加权的池化
                for i in range(self.config.seq_length):
                    # [batch, w, dim, 1], [batch, w, 1, 1] => [batch, 1, dim, 1]
                    kernel_size = self.config.kernel_size
                    pools.append(tf.reduce_sum(x[:, i:i + kernel_size, :, :] * attention[:, i:i + kernel_size, :, :],
                                               axis=1,
                                               keep_dims=True))

                # [batch, seq_length, dim, 1]
                w_ap = tf.concat(pools, axis=1, name="w_ap")
            else:
                # 平均池化，[batch, seq_length, dim]
                w_ap = tf.layers.average_pooling2d(inputs=x,
                                                   pool_size=(self.config.kernel_size, 1),
                                                   strides=1,
                                                   padding="VALID",
                                                   name="w_ap")
            return w_ap


    def all_pool(self, variable_scope, x):
        '''
        全局池化函数
        :param variable_scope: 变量空间
        :param x: 输入向量 [batch_size, seq_length, hidden_dim, 1]
        :return:
        '''
        with tf.variable_scope(variable_scope + "-all_pool"):
            # 如果是对初始input进行all-pool
            if variable_scope.startswith("input"):
                pool_width = self.config.seq_length     # 文本长度
                d = self.config.embedding_dim           # 词向量维度

            else:    # 如果是对中间卷积结果进行all-pool
                pool_width = self.config.seq_length + self.config.kernel_size - 1
                d = self.config.hidden_dim

            # 二维平均池化
            all_ap = tf.layers.average_pooling2d(inputs=x,
                                                 pool_size=(pool_width, 1),
                                                 strides=1,
                                                 padding='VALID',
                                                 name='all_ap')
            # [batch, hidden_dim]
            all_ap_reshaped = tf.reshape(all_ap, [-1, d])

            return all_ap_reshaped


    def CNN_layer(self, variable_scope, query, doc, dim):
        '''
        卷积层 pad + conv + pool
        :param variable_scope: 变量空间
        :param query: 输入的query向量
        :param doc: 输入的doc向量
        :param dim: 卷积核宽度（词向量大小 or 隐藏层向量大小）
        :return:
        '''
        # x1, x2 = [batch, seq_length, embedding_dim, 1]    dim:hidden_dim
        model_type = self.config.model_type
        with tf.variable_scope(variable_scope):
            if model_type == "ABCNN1" or model_type == "ABCNN3":
                with tf.name_scope("att_mat"):
                    aW = tf.get_variable(name="aW",
                                         shape=(self.config.seq_length, dim),
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         regularizer=tf.contrib.layers.l2_regularizer(scale=self.config.l2_reg))

                    # attention weight [batch, seq_length, seq_length]
                    att_mat = self.make_attention_mat(query, doc)
                    query_a = tf.expand_dims(tf.einsum("ijk,kl->ijl", att_mat, aW), axis=-1)    # attention交互生成的新query embedding
                    doc_a = tf.expand_dims(tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat), aW), axis=-1) # attention交互生成的新doc embedding
                    # [batch, d, s, 2]
                    query = tf.concat([query, query_a], axis=3)     # 新embedding与旧embedding在第三维度上进行组合拼接
                    doc = tf.concat([doc, doc_a], axis=3)

            # 进行pad + 宽卷积
            left_conv = self.convolution(name_scope="left", x=self.pad_for_wide_conv(query), dim=dim, reuse=False)
            right_conv = self.convolution(name_scope="right", x=self.pad_for_wide_conv(doc), dim=dim, reuse=True)

            left_attention, right_attention = None, None

            if model_type == "ABCNN2" or model_type == "ABCNN3":
                # [batch, s+w-1, s+w-1]
                att_mat = self.make_attention_mat(left_conv, right_conv)
                # 获取left和right的attention权重进行加权的池化操作，[batch, s+w-1], [batch, s+w-1]
                left_attention, right_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)

            # 进行池化处理
            left_wp = self.w_pool(variable_scope="left", x=left_conv, attention=left_attention)
            left_ap = self.all_pool(variable_scope="left", x=left_conv)
            right_wp = self.w_pool(variable_scope="right", x=right_conv, attention=right_attention)
            right_ap = self.all_pool(variable_scope="right", x=right_conv)

            return left_wp, left_ap, right_wp, right_ap



if __name__ == '__main__':
    Model()