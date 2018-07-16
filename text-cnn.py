import os
import sys
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Input,Flatten
from keras.layers import Conv1D,MaxPooling1D,Embedding
from keras.models import Model
from keras.utils.np_utils import to_categorical

TEXT_DATA_DIR='glove.6B.100d.txt'
MAX_SEQUENCE_LENGTH=1000
MAX_NB_WORDS=10000
EMBEDDING_DIM=100
VALIDATION_SPLIT=0.2

# 建立word与向量的索引
embeddings_index={}
f=open(TEXT_DATA_DIR,'r',encoding='utf-8')
for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:],dtype='float32')
    embeddings_index[word]=coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

# 选定的类别
selected_categories = [
    'comp.graphics',
    'rec.motorcycles',
    'rec.sport.baseball',
    'misc.forsale',
    'sci.electronics',
    'sci.med',
    'talk.politics.guns',
    'talk.religion.misc']

# 加载数据
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
print(type(texts[0]),texts)

# 文本向量化
tokenizer=Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences=tokenizer.texts_to_sequences(texts)

word_index=tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_NB_WORDS)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
print('Data validation split.')

# 将data划分为训练集和验证集
indices=np.arange(data.shape[0])
np.random.shuffle(indices)
data=data[indices]
labels=labels[indices]
num_validation_samples=int(VALIDATION_SPLIT*data.shape[0])
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


# 创建词嵌入
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # 如果单词没有在word index,全部设置为0.
        embedding_matrix[i] = embedding_vector
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_NB_WORDS,
                            trainable=False)

print("正在初始化模型...")

# 初始化卷积层和池化层
sequence_input=Input(shape=(MAX_NB_WORDS,),dtype='int32')
embedded_sequences=embedding_layer(sequence_input)
x=Conv1D(128,5,activation='relu')(embedded_sequences)
x=MaxPooling1D(5)(x)
x=Conv1D(128,5,activation='relu')(x)
x=MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(8, activation='softmax')(x)

model=Model(sequence_input,preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
print("正在训练模型...")
# 训练模型
model.fit(x_train,y_train,
          batch_size=128,
          nb_epoch=10,
          validation_data=(x_val,y_val))