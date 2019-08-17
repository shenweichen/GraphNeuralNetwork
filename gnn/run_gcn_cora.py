#!/usr/bin/env python
# coding: utf-8

import numpy as np
from gcn import GCN
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Lambda
from utils import load_data, get_splits, preprocess_adj,plot_embeddings

if __name__ == "__main__":

    FEATURE_LESS = False
    features, A, y, _ = load_data(dataset="cora")
    y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, val_mask, test_mask = get_splits(y,
                                                                                                       shuffle=False)
    features /= features.sum(axis=1, ).reshape(-1, 1)

    A = preprocess_adj(A)

    if FEATURE_LESS:
        X = np.arange(A.shape[-1])
        feature_dim = A.shape[-1]
    else:
        X = features
        feature_dim = X.shape[-1]
    model_input = [X, A]

    # Compile model
    model = GCN(A.shape[-1], y_train.shape[1], feature_dim, dropout_rate=0.5, l2_reg=2.5e-4,
                feature_less=FEATURE_LESS, )
    model.compile(optimizer=Adam(0.01), loss='categorical_crossentropy',
                  weighted_metrics=['categorical_crossentropy', 'acc'])

    NB_EPOCH = 300
    PATIENCE = 10  # early stopping patience

    val_data = (model_input, y_val, val_mask)
    es_callback = EarlyStopping(monitor='weighted_categorical_crossentropy', patience=PATIENCE)

    # train
    print("start training")
    model.fit(model_input, y_train, sample_weight=train_mask, validation_data=val_data,
              batch_size=A.shape[0], epochs=NB_EPOCH, shuffle=False, verbose=2, callbacks=[es_callback])
    # test
    eval_results = model.evaluate(model_input, y_test, sample_weight=test_mask, batch_size=A.shape[0])
    print('Done.\n'
          'Test loss: {}\n'
          'Test weighted_loss: {}\n'
          'Test accuracy: {}'.format(*eval_results))

    embedding_model = Model(model.input, outputs=Lambda(lambda x: model.layers[-1].output)(model.input))
    embedding_weights = embedding_model.predict(model_input, batch_size=A.shape[0])
    y  = np.genfromtxt("{}{}.content".format('../data/cora/', 'cora'), dtype=np.dtype(str))[:, -1]
    plot_embeddings(embedding_weights, np.arange(A.shape[0]), y)
