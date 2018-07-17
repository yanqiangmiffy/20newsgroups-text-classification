from __future__ import print_function

import os
import sys
import numpy as np
from pprint import pprint

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.utils.np_utils import to_categorical

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from cnn_lstm import CNNLSTM
from sklearn.externals import joblib

selected_categories = [
    'comp.graphics',
    'comp.windows.x',
    'rec.motorcycles',
    'rec.sport.baseball',
    'sci.crypt',
    'sci.med',
    'talk.politics.guns',
    'talk.religion.misc']

newsgroups_train = fetch_20newsgroups(subset='train',
                                      categories=selected_categories,
                                      remove=('headers', 'footers', 'quotes'))

newsgroups_test = fetch_20newsgroups(subset='test',
                                     categories=selected_categories,
                                     remove=('headers', 'footers', 'quotes'))
texts = newsgroups_train['data']
labels = newsgroups_train['target']

print(len(texts))
print(np.unique(labels))
print(labels)

texts = [t for t in texts]
print(type(texts[0]))

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


ax_features = 10000
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

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
#
print('Initialize model.')

model = CNNLSTM().initialize()

print('x_train shape:', x_train.shape)
print('x_test shape:', x_val.shape)


print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs,
          validation_data=(x_val, y_val), verbose=1)
score, acc = model.evaluate(x_val, y_val, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

logreg = linear_model.LogisticRegression(C=1e5, verbose=1)

# we create an instance of Neighbours Classifier and fit the data.
print('Training')

logreg.fit(x_train, y_train)
joblib.dump(logreg, 'logreg.pkl')


clf = joblib.load('logreg.pkl')

pred = clf.predict(x_train)
print(((pred == y_train).sum() * 100.0) / (np.shape(y_train)[0] * 1.0))


# from sklearn import datasets
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(x_train, y_train)
# joblib.dump(gnb, 'gnb.pkl')
# clf = joblib.load('gnb.pkl')
# pred = clf.predict(x_train)
# print(((pred == y_train).sum() * 100.0) / (np.shape(y_train)[0] * 1.0))