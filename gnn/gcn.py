#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""



Author:

    Weichen Shen,wcshen1994@163.com



Reference:

    [1] Kipf T N, Welling M. Semi-supervised classification with graph convolutional networks[J]. arXiv preprint arXiv:1609.02907, 2016.(https://arxiv.org/pdf/1609.02907)



"""

import tensorflow as tf
from tensorflow.python.keras.initializers import Identity, glorot_uniform, Zeros
from tensorflow.python.keras.layers import Dropout, Input, Layer, Embedding, Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2


class GraphConvolution(Layer):  # ReLU(AXW)

    def __init__(self, units,
                 activation=tf.nn.relu, dropout_rate=0.5,
                 use_bias=True, l2_reg=0, feature_less=False,
                 seed=1024, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.feature_less = feature_less
        self.use_bias = use_bias
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.seed = seed

    def build(self, input_shapes):

        if self.feature_less:
            input_dim = int(input_shapes[0][-1])
        else:
            assert len(input_shapes) == 2
            features_shape = input_shapes[0]

            input_dim = int(features_shape[-1])

        self.kernel = self.add_weight(shape=(input_dim,
                                             self.units),
                                      initializer=glorot_uniform(
                                          seed=self.seed),
                                      regularizer=l2(self.l2_reg),
                                      name='kernel', )
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=Zeros(),
                                        name='bias', )

        self.dropout = Dropout(self.dropout_rate, seed=self.seed)

        self.built = True

    def call(self, inputs, training=None, **kwargs):
        features, A = inputs
        features = self.dropout(features, training=training)
        output = tf.matmul(tf.sparse_tensor_dense_matmul(
            A, features), self.kernel)
        if self.use_bias:
            output += self.bias
        act = self.activation(output)

        act._uses_learning_phase = features._uses_learning_phase
        return act

    def get_config(self):
        config = {'units': self.units,
                  'activation': self.activation,
                  'dropout_rate': self.dropout_rate,
                  'l2_reg': self.l2_reg,
                  'use_bias': self.use_bias,
                  'feature_less': self.feature_less,
                  'seed': self.seed
                  }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def GCN(adj_dim,feature_dim,n_hidden, num_class, num_layers=2,activation=tf.nn.relu,dropout_rate=0.5, l2_reg=0, feature_less=True, ):
    Adj = Input(shape=(None,), sparse=True)
    if feature_less:
        X_in = Input(shape=(1,), )

        emb = Embedding(adj_dim, feature_dim,
                        embeddings_initializer=Identity(1.0), trainable=False)
        X_emb = emb(X_in)
        h = Reshape([X_emb.shape[-1]])(X_emb)
    else:
        X_in = Input(shape=(feature_dim,), )

        h = X_in

    for i in range(num_layers):
        if i == num_layers - 1:
            activation = tf.nn.softmax
            n_hidden = num_class
        h = GraphConvolution(n_hidden, activation=activation, dropout_rate=dropout_rate, l2_reg=l2_reg)([h,Adj])

    output = h
    model = Model(inputs=[X_in,Adj], outputs=output)

    return model
