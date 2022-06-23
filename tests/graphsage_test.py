#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import numpy as np
import tensorflow as tf

try:
    from tensorflow.python.keras.optimizers import Adam
except ImportError:
    from tensorflow.keras.optimizers import Adam

from gnn.graphsage import sample_neighs, GraphSAGE
from gnn.utils import preprocess_adj, load_data_v1


def test_GraphSAGE():
    # Read data

    A, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_v1(
        'cora', path="./data/cora/")

    features /= features.sum(axis=1, ).reshape(-1, 1)

    G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())

    A = preprocess_adj(A)

    indexs = np.arange(A.shape[0])
    neigh_number = [10, 25]
    neigh_maxlen = []

    model_input = [features, np.asarray(indexs, dtype=np.int32)]

    for num in neigh_number:
        sample_neigh, sample_neigh_len = sample_neighs(
            G, indexs, num, self_loop=False)
        model_input.extend([sample_neigh])
        neigh_maxlen.append(max(sample_neigh_len))

    model = GraphSAGE(feature_dim=features.shape[1],
                      neighbor_num=neigh_maxlen,
                      n_hidden=16,
                      n_classes=y_train.shape[1],
                      use_bias=True,
                      activation=tf.nn.relu,
                      aggregator_type='mean',
                      dropout_rate=0.5, l2_reg=2.5e-4)
    model.compile(Adam(0.01), 'categorical_crossentropy',
                  weighted_metrics=['categorical_crossentropy', 'acc'])

    val_data = (model_input, y_val, val_mask)
    # mc_callback = ModelCheckpoint('./best_model.h5',
    #                               monitor='val_weighted_categorical_crossentropy',
    #                               save_best_only=True,
    #                               save_weights_only=True)

    print("start training")
    model.fit(model_input, y_train, sample_weight=train_mask, validation_data=val_data,
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=2,
              callbacks=[])


if __name__ == "__main__":
    pass
