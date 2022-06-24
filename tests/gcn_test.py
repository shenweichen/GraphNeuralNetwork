#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pytest

try:
    from tensorflow.python.keras.optimizers import Adam
except ImportError:
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    from tensorflow.python.keras.optimizer_v1 import Adam

from gnn.gcn import GCN
from gnn.utils import preprocess_adj, load_data_v1


@pytest.mark.parametrize(
    'FEATURE_LESS',
    [True, False
     ]
)
def test_GCN(FEATURE_LESS):
    A, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_v1(
        'cora', path="./data/cora/")

    A = preprocess_adj(A)

    features /= features.sum(axis=1, ).reshape(-1, 1)

    if FEATURE_LESS:
        X = np.arange(A.shape[-1])
        feature_dim = A.shape[-1]
    else:
        X = features
        feature_dim = X.shape[-1]
    model_input = [X, A]

    # Compile model
    model = GCN(A.shape[-1], feature_dim, 16, y_train.shape[1], dropout_rate=0.5, l2_reg=2.5e-4,
                feature_less=FEATURE_LESS, )
    model.compile(optimizer=Adam(0.01), loss='categorical_crossentropy',
                  weighted_metrics=['categorical_crossentropy', 'acc'])

    NB_EPOCH = 1
    PATIENCE = 200  # early stopping patience

    val_data = (model_input, y_val, val_mask)
    # mc_callback = ModelCheckpoint('./best_model.h5',
    #                               monitor='val_weighted_categorical_crossentropy',
    #                               save_best_only=True,
    #                               save_weights_only=True)

    # train
    print("start training")
    model.fit(model_input, y_train, sample_weight=train_mask, validation_data=val_data,
              batch_size=A.shape[0], epochs=NB_EPOCH, shuffle=False, verbose=2, callbacks=[])


if __name__ == "__main__":
    pass
