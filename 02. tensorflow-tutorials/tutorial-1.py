# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:23:39 2020

@author: -
"""

import tensorflow as tf

# mnist 데이터셋 
mnist = tf.keras.datasets.mnist

# 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 모델 생성
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])


# 모델 컴파일(sparse => 정수 형태의 class)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train, epochs=5)

# 모델 저장
model.save_weights("./models/tutorial-1/model.ckpt")

# 모델 평가
model.evaluate(x_test, y_test, verbose=2)