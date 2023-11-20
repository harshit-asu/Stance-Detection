import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding

def train(X, y):
    model = Sequential()
    # model.add(Embedding(input_dim=768, output_dim=64, input_length=len(X[0])))
    model.add(LSTM(64, input_shape=(len(X[0]),1), return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation= 'softmax'))
    model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
    model.fit(X, y, epochs=10)
    return model