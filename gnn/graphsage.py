#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""



Author:

    Weichen Shen,wcshen1994@163.com



Reference:

    [1] Hamilton W, Ying Z, Leskovec J. Inductive representation learning on large graphs[C]//Advances in Neural Information Processing Systems. 2017: 1024-1034.(https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)



"""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.initializers import glorot_uniform, Zeros
from tensorflow.python.keras.layers import Input, Dense, Dropout, Layer, LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2


class MeanAggregator(Layer):

    def __init__(self, units, input_dim, neigh_max, concat=True, dropout_rate=0.0, activation=tf.nn.relu, l2_reg=0,
                 use_bias=False,
                 seed=1024, **kwargs):
        super(MeanAggregator, self).__init__()
        self.units = units
        self.neigh_max = neigh_max
        self.concat = concat
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.activation = activation
        self.seed = seed
        self.input_dim = input_dim

    def build(self, input_shapes):

        self.neigh_weights = self.add_weight(shape=(self.input_dim, self.units),
                                             initializer=glorot_uniform(
                                                 seed=self.seed),
                                             regularizer=l2(self.l2_reg),
                                             name="neigh_weights")
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units), initializer=Zeros(),
                                        name='bias_weight')

        self.dropout = Dropout(self.dropout_rate)
        self.built = True

    def call(self, inputs, training=None):
        features, node, neighbours = inputs

        node_feat = tf.nn.embedding_lookup(features, node)
        neigh_feat = tf.nn.embedding_lookup(features, neighbours)

        node_feat = self.dropout(node_feat, training=training)
        neigh_feat = self.dropout(neigh_feat, training=training)

        concat_feat = tf.concat([neigh_feat, node_feat], axis=1)
        concat_mean = tf.reduce_mean(concat_feat, axis=1, keep_dims=False)

        output = tf.matmul(concat_mean, self.neigh_weights)
        if self.use_bias:
            output += self.bias
        if self.activation:
            output = self.activation(output)

        # output = tf.nn.l2_normalize(output,dim=-1)
        output._uses_learning_phase = True

        return output

    def get_config(self):
        config = {'units': self.units,
                  'concat': self.concat,
                  'seed': self.seed
                  }

        base_config = super(MeanAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoolingAggregator(Layer):

    def __init__(self, units, input_dim, neigh_max, aggregator='meanpooling', concat=True,
                 dropout_rate=0.0,
                 activation=tf.nn.relu, l2_reg=0, use_bias=False,
                 seed=1024, ):
        super(PoolingAggregator, self).__init__()
        self.output_dim = units
        self.input_dim = input_dim
        self.concat = concat
        self.pooling = aggregator
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.activation = activation
        self.neigh_max = neigh_max
        self.seed = seed

        # if neigh_input_dim is None:

    def build(self, input_shapes):

        self.dense_layers = [Dense(
            self.input_dim, activation='relu', use_bias=True, kernel_regularizer=l2(self.l2_reg))]

        self.neigh_weights = self.add_weight(
            shape=(self.input_dim * 2, self.output_dim),
            initializer=glorot_uniform(
                seed=self.seed),
            regularizer=l2(self.l2_reg),

            name="neigh_weights")

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=Zeros(),
                                        name='bias_weight')

        self.built = True

    def call(self, inputs, mask=None):

        features, node, neighbours = inputs

        node_feat = tf.nn.embedding_lookup(features, node)
        neigh_feat = tf.nn.embedding_lookup(features, neighbours)

        dims = tf.shape(neigh_feat)
        batch_size = dims[0]
        num_neighbors = dims[1]
        h_reshaped = tf.reshape(
            neigh_feat, (batch_size * num_neighbors, self.input_dim))

        for l in self.dense_layers:
            h_reshaped = l(h_reshaped)
        neigh_feat = tf.reshape(
            h_reshaped, (batch_size, num_neighbors, int(h_reshaped.shape[-1])))

        if self.pooling == "meanpooling":
            neigh_feat = tf.reduce_mean(neigh_feat, axis=1, keep_dims=False)
        else:
            neigh_feat = tf.reduce_max(neigh_feat, axis=1)

        output = tf.concat(
            [tf.squeeze(node_feat, axis=1), neigh_feat], axis=-1)

        output = tf.matmul(output, self.neigh_weights)
        if self.use_bias:
            output += self.bias
        if self.activation:
            output = self.activation(output)

        # output = tf.nn.l2_normalize(output, dim=-1)

        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'concat': self.concat
                  }

        base_config = super(PoolingAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTMAggregator(Layer):

    def __init__(self, units, input_dim, activation=tf.nn.relu, concat=True, dropout_rate=0.0, l2_reg=0, use_bias=False,
                 seed=1024, **kwargs):
        super(LSTMAggregator, self).__init__()
        self.units = units
        self.input_dim = input_dim
        self.concat = concat
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.activation = activation
        self.seed = seed

    def build(self, input_shapes):

        # self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
        #                                     name='neigh_weights')
        #
        # self.vars['self_weights'] = glorot([input_dim, output_dim],
        #                                    name='self_weights')

        self.node_weights = self.add_weight(shape=(self.input_dim, self.units),
                                            initializer=glorot_uniform(
                                                seed=self.seed),
                                            regularizer=l2(self.l2_reg),
                                            name="node_weights")

        self.neigh_weights = self.add_weight(shape=(self.units, self.units),
                                             initializer=glorot_uniform(
                                                 seed=self.seed),
                                             regularizer=l2(self.l2_reg),
                                             name="neigh_weights")

        # self.temp_weights = self.add_weight(
        #     shape=(self.units * 2, self.units),
        #     initializer=glorot_uniform(
        #         seed=self.seed),
        #     regularizer=l2(self.l2_reg),
        #
        #     name="temp_weights")
        # self.dropout = Dropout(self.dropout_rate)
        self.lstm = LSTM(self.units, dropout=0.0, bias_initializer='ones', return_sequences=False,
                         unroll=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,), initializer=Zeros(),
                                        name='bias_weight')

        self.built = True

    def call(self, inputs, training=None):  # tf.nn.dropout(neigh_feat,1-self.dropout_rate)

        features, node, neighbours = inputs

        node_feat = tf.nn.embedding_lookup(features, node)
        neigh_feat = tf.nn.embedding_lookup(features, neighbours)

        neigh_means = self.lstm(neigh_feat)

        from_self = tf.matmul(tf.squeeze(node_feat, axis=1), self.node_weights)
        # [?,10,128], [1433,128].
        from_neighs = neigh_means  # tf.matmul(neigh_means, self.neigh_weights)

        # output = tf.concat(
        #    [tf.squeeze(node_feat, axis=1), neigh_means], axis=-1)
        # output = tf.matmul(output, self.neigh_weights)  # ?,1449], [32,16].

        # if self.concat:
        # output = #tf.concat([from_self, from_neighs], axis=1)
        output = (from_self + from_neighs) / 2  # tf.reduce_mean(output,axis=1)
        # else:
        # output = tf.add_n([from_self, from_neighs])
        # output = tf.matmul(output,self.temp_weights)

        if self.use_bias:
            output += self.bias

        if self.activation:
            output = self.activation(output)

        output._uses_learning_phase = True

        return output

    def get_config(self):
        config = {'units': self.units,
                  'concat': self.concat,
                  'seed': self.seed
                  }

        base_config = super(LSTMAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def GraphSAGE(feature_dim, neighbor_num, n_hidden, n_classes, use_bias=True, activation=tf.nn.relu,
              aggregator_type='mean', dropout_rate=0.0, l2_reg=0):
    features = Input(shape=(feature_dim,))
    node_input = Input(shape=(1,), dtype=tf.int32)
    neighbor_input = [Input(shape=(l,), dtype=tf.int32) for l in neighbor_num]

    if aggregator_type == 'mean':
        aggregator = MeanAggregator
    # elif aggregator_type == 'lstm':
    #     aggregator = LSTMAggregator
    else:
        aggregator = PoolingAggregator

    h = features
    for i in range(0, len(neighbor_num)):
        if i > 0:
            feature_dim = n_hidden
        if i == len(neighbor_num) - 1:
            activation = tf.nn.softmax
            n_hidden = n_classes
        h = aggregator(units=n_hidden, input_dim=feature_dim, activation=activation, l2_reg=l2_reg, use_bias=use_bias,
                       dropout_rate=dropout_rate, neigh_max=neighbor_num[i], aggregator=aggregator_type)(
            [h, node_input, neighbor_input[i]])  #

    output = h
    input_list = [features, node_input] + neighbor_input
    model = Model(input_list, outputs=output)
    return model


def sample_neighs(G, nodes, sample_num=None, self_loop=False, shuffle=True):  # 抽样邻居节点
    _sample = np.random.choice
    neighs = [list(G[int(node)]) for node in nodes]  # nodes里每个节点的邻居
    if sample_num:
        if self_loop:
            sample_num -= 1

        samp_neighs = [
            list(_sample(neigh, sample_num, replace=False)) if len(neigh) >= sample_num else list(
                _sample(neigh, sample_num, replace=True)) for neigh in neighs]  # 采样邻居
        if self_loop:
            samp_neighs = [
                samp_neigh + list([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]  # gcn邻居要加上自己

        if shuffle:
            samp_neighs = [list(np.random.permutation(x)) for x in samp_neighs]
    else:
        samp_neighs = neighs
    return np.asarray(samp_neighs), np.asarray(list(map(len, samp_neighs)))
