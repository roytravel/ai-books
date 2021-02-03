# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print ('버전: ', tf.__version__)
print ('즉시 실행 모드: ', tf.executing_eagerly())
print ("허브 버전: ", hub.__version__)
print ("GPU 사용 가능: " if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# IMDB 데이터셋 다운로드
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)


# 데이터 탐색
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

# print ('[+] Train examples batch : ',train_examples_batch)
# print ('[+] Train labels batch : ', train_labels_batch )

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

# 모델 생성
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

# print (model.summary())

# 손실 함수와 옵티마이저
model.compile(optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])

# 모델 훈련
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# 모델 저장
model.save_weights('./models/tutorial-4/model.ckpt')


# 모델 평가
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))