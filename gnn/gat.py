from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Zeros
from tensorflow.python.keras.layers import Layer, Dropout,Input
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Model


class GATLayer(Layer):

    def __init__(self, att_embedding_size=8, head_num=8, dropout_rate=0.5, l2_reg=0, activation=tf.nn.relu,
                 reduction='concat', use_bias=True, seed=1024, **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.activation = activation
        self.act = activation
        self.reduction = reduction
        self.use_bias = use_bias
        self.seed = seed
        super(GATLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        X, A = input_shape
        embedding_size = int(X[-1])
        self.weight = self.add_weight(name='weight', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                      dtype=tf.float32,
                                      regularizer=l2(self.l2_reg),
                                      initializer=tf.keras.initializers.glorot_uniform())
        self.att_self_weight = self.add_weight(name='att_self_weight',
                                               shape=[1, self.head_num,
                                                      self.att_embedding_size],
                                               dtype=tf.float32,
                                               regularizer=l2(self.l2_reg),
                                               initializer=tf.keras.initializers.glorot_uniform())
        self.att_neighs_weight = self.add_weight(name='att_neighs_weight',
                                                 shape=[1, self.head_num,
                                                        self.att_embedding_size],
                                                 dtype=tf.float32,
                                                 regularizer=l2(self.l2_reg),
                                                 initializer=tf.keras.initializers.glorot_uniform())

        if self.use_bias:
            self.bias_weight = self.add_weight(name='bias', shape=[1, self.head_num, self.att_embedding_size],
                                               dtype=tf.float32,
                                               initializer=Zeros())
        self.in_dropout = Dropout(self.dropout_rate)
        self.feat_dropout = Dropout(self.dropout_rate, )
        self.att_dropout = Dropout(self.dropout_rate, )
        # Be sure to call this somewhere!
        super(GATLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):

        X, A = inputs
        X = self.in_dropout(X)  # N * D
        # A = self.att_dropout(A, training=training)
        if K.ndim(X) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(X)))

        features = tf.matmul(X, self.weight, )  # None F'*head_num
        features = tf.reshape(
            features, [-1, self.head_num, self.att_embedding_size])  # None head_num F'

        # attn_for_self = K.dot(features, attention_kernel[0])  # (N x 1), [a_1]^T [Wh_i]
        # attn_for_neighs = K.dot(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

        # head_num None F D --- > head_num None(F) D

        # querys = tf.stack(tf.split(querys, self.head_num, axis=1))
        # keys = tf.stack(tf.split(keys, self.head_num, axis=1))#[?,1,1433,64]

        # features = tf.stack(tf.split(features, self.head_num, axis=1))  # head_num None F'
        attn_for_self = tf.reduce_sum(
            features * self.att_self_weight, axis=-1, keep_dims=True)  # None head_num 1
        attn_for_neighs = tf.reduce_sum(
            features * self.att_neighs_weight, axis=-1, keep_dims=True)
        dense = tf.transpose(
            attn_for_self, [1, 0, 2]) + tf.transpose(attn_for_neighs, [1, 2, 0])

        dense = tf.nn.leaky_relu(dense, alpha=0.2)
        mask = -10e9 * (1.0 - A)
        dense += tf.expand_dims(mask, axis=0)  # [?,8,8], [1,?,2708]

        self.normalized_att_scores = tf.nn.softmax(
            dense, dim=-1, )  # head_num None(F) None(F)

        features = self.feat_dropout(features, )
        self.normalized_att_scores = self.att_dropout(
            self.normalized_att_scores)

        result = tf.matmul(self.normalized_att_scores,
                           tf.transpose(features, [1, 0, 2]))  # head_num None F D   [8,2708,8] [8,2708,3]
        result = tf.transpose(result, [1, 0, 2])  # None head_num attsize

        if self.use_bias:
            result += self.bias_weight

        # head_num Node embeding_size
        if self.reduction == "concat":
            result = tf.concat(
                tf.split(result, self.head_num, axis=1), axis=-1)
            result = tf.squeeze(result, axis=1)
        else:
            result = tf.reduce_mean(result, axis=1)

        if self.act:
            result = self.activation(result)

        result._uses_learning_phase = True
        return result

    def compute_output_shape(self, input_shape):
        if self.reduction == "concat":

            return (None, self.att_embedding_size * self.head_num)
        else:
            return (None, self.att_embedding_size)

    def get_config(self, ):
        config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num, 'use_res': self.use_res,
                  'seed': self.seed}
        base_config = super(GATLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




def GAT(adj_dim,feature_dim,num_class,num_layers=2,n_attn_heads = 8,att_embedding_size=8,dropout_rate=0.0,l2_reg=0.0,use_bias=True):
    X_in = Input(shape=(feature_dim,))
    A_in = Input(shape=(adj_dim,))
    h = X_in
    for _ in range(num_layers-1):
        h = GATLayer(att_embedding_size=att_embedding_size, head_num=n_attn_heads, dropout_rate=dropout_rate, l2_reg=l2_reg,
                                     activation=tf.nn.elu, use_bias=use_bias, )([h, A_in])

    h = GATLayer(att_embedding_size=num_class, head_num=1, dropout_rate=dropout_rate, l2_reg=l2_reg,
                                 activation=tf.nn.softmax, use_bias=use_bias, reduction='mean')([h, A_in])

    model = Model(inputs=[X_in, A_in], outputs=h)

    return model