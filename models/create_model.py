import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Flatten
from tensorflow.keras.models import Sequential


# Basic Neural Network
def NN_BASE(data, metrics):
    model = Sequential()
    model.add(Embedding(50000, 100, input_length=data.shape[1]))
    model.add(Dense(128,activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    return model

# Neural Network with a Dense Layer and LSTM layer with dropout
def NN_LSTM_DROPOUT(data, metrics):
    model = Sequential()
    model.add(Embedding(50000, 100, input_length=data.shape[1]))
    model.add(Dense(128,activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    return model

# Neural Network with a LSTM layer
def NN_LSTM(data, metrics):
    model = Sequential()
    model.add(Embedding(50000, 100, input_length=data.shape[1]))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    return model

