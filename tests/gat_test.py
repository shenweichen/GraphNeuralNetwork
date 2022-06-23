#!/usr/bin/env python
# coding: utf-8

import pytest
import scipy.sparse as sp
import  numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from gnn.gat import GAT
from gnn.utils import preprocess_adj, load_data_v1


def test_GAT():
    # Read data

    A, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_v1(
        'cora', path="./data/cora/")

    A = A + sp.eye(A.shape[0])
    features /= features.sum(axis=1, ).reshape(-1, 1)

    model = GAT(adj_dim=A.shape[0], feature_dim=features.shape[1], num_class=y_train.shape[1], num_layers=2,
                n_attn_heads=8, att_embedding_size=8,
                dropout_rate=0.6, l2_reg=2.5e-4, use_bias=True)
    optimizer = Adam(lr=0.005)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  weighted_metrics=['categorical_crossentropy', 'acc'])

    model_input = [features, A.toarray()]

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
