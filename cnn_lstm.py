'''Train a recurrent convolutional network on the IMDB sentiment
classification task.
Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb


class CNNLSTM():
    def __init__(self):
        pass
    # Embedding
    max_features = 10000
    maxlen = 1000
    embedding_size = 128

    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4

    # LSTM
    lstm_output_size = 70

    # Training
    batch_size = 30
    epochs = 2

    '''
    Note:
    batch_size is highly sensitive.
    Only 2 epochs are needed as the dataset is very small.
    '''
    def initialize(self):
        print('Build model...')
        model = Sequential()
        model.add(Embedding(self.max_features, self.embedding_size, input_length=self.maxlen))
        model.add(Dropout(0.25))
        model.add(Conv1D(self.filters,
                         self.kernel_size,
                         border_mode='valid',
                         activation='relu'))
        model.add(MaxPooling1D(pool_length=self.pool_size))
        model.add(LSTM(self.lstm_output_size))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        print('Model Compiled')
        return model

    def train(self, model):
        print('Loading data...')
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_features)
        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')

        print('Pad sequences (samples x time)')
        x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)


        print('Train...')
        model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test))
        score, acc = model.evaluate(x_test, y_test, batch_size=self.batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

model = CNNLSTM().initialize()