import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def train_lstm(X_train, y_train, X_val, y_val, input_shape, epochs=100, threads=-1):
    """
    X_train, y_train: training data
    X_val, y_val: validation data
    input_shape: (sequence_length, num_features)
    """

    if threads != -1:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(threads)

    # 1. Functional Keras Model
    sequence_length, num_features = input_shape

    # Define Inputs
    inputs = Input(shape=(sequence_length, num_features))

    # LSTM layers
    x = LSTM(50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(50)(x)
    x = Dropout(0.2)(x)

    # Output layer
    outputs = Dense(1)(x)

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 2. Fit the model
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    return model
