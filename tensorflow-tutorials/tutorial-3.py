# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 21:54:41 2020

@author: -
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print (tf.__version__)

# 데이터셋 다운로드
imdb = keras.datasets.imdb

# 상위 10,000개 단어 선택
(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words=10000)

# 데이터 탐색
print ('Training sample : {}\nLabel sample : {}'.format(len(train_data), len(train_label)))

# print (train_data[0])
# print (train_label[0])

# 단어와 정수 인덱스를 매핑한 딕셔너리
word_index = imdb.get_word_index()


word_index = {k:(v+3) for k,v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


#print (decode_review(train_data[0]))

# 데이터 준비
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                      value=word_index["<PAD>"],
                                                      padding='post',
                                                      maxlen=256)
# 패딩 확인                                                      
print (len(train_data[0]), len(train_data[1]))
print (train_data[0])

# 모델 구성

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None, )))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 검증 셋 생성
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_label[:10000]
partial_y_train = train_label[10000:]

# 모델 훈련
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# 모델 저장
model.save_weights('./models/tutorial-3/model.ckpt')

# 모델 평가
results = model.evaluate(test_data, test_label, verbose=2)
print (results)

# 정확도와 손실 그래프 작성
history_dict = history.history
print (history_dict.keys())