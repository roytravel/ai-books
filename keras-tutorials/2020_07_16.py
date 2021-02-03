# -*- coding:utf-8 -*-
# Sequential 클래스 사용하여 정의한 2개의 층으로 된 모델
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
model.add(layers.Dense(10, activation='softmax'))

# 같은 모델을 함수형 API를 사용한 모습
input_tensor = layers.Input(shape=(784))
x = layers.Dense(32, activation='relu')(input_tensor)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = models.Model(input=input_tensor, output=output_tensor)

# 학습 과정이 설정되는 컴파일 단계
from keras import optimizers

model.compile(optimizers=optimizers.RMSprop(lr=0.001),loss = 'mse',metrics = ['accuracy'])

model.fit(input_tensor, output_tensor, batch_size=128, epoch=10)
