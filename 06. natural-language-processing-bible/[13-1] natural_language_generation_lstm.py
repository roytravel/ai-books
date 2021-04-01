# -*- coding:utf-8 -*-
"""
Title
    - Natural language generation using LSTM network
Purpose
    - Learn a series of process like loading the train dataset, define the LSTM network structure and learn the network using deep learning.
"""


import io
import sys
# from __future__ import print_function
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import get_file
import numpy as np 
import random
import sys
import io

    
def get_text_file():
    """ load the text file """
    fpath = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    with io.open(fpath, encoding='utf-8') as f:
        text = f.read().lower()
    
    return text


def create_vocabulary_dictionary(text):
    """ create vocabulary dictionary """
    chars = sorted(list(set(text)))
    char2index = dict((c, i) for i, c in enumerate(chars))
    index2char = dict((i, c) for i, c in enumerate(chars))

    return chars, char2index, index2char


def create_syllable(text, chars, char2index):
    # Generate syllable learning data
    maxlen, step = 40, 3
    sentences, next_chars = [], []

    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i:i+maxlen])
        next_chars.append(text[i+maxlen])
    
    print ("The number of sentences:", len(sentences))
    
    # np zero function return the ndarray of shape dimension initialized by 0
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    # ???
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char2index[char]] = 1
        y[i, char2index[next_chars[i]]] = 1

    return x, y, maxlen


def create_model(maxlen, chars):
    """ Declaring a deep learning model """
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
    return model

""" next syllable sampling upon input probability """

def sample(preds, temperature=1.0):
    """np.asarray is same with array but, array basically copy=true is default and asarray basically copy=false is default.
    It used in case of creating ndarray from inputting a list"""
    preds = np.asarray(preds).astype('float64')

    """ np.log is calculate the all elements of natural logarithm """
    preds = np.log(preds) / temperature

    """ np.exp return an exponential value """
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    """ np.random.multinomial simulate the multinomial distribution """
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)





def main():

    text = get_text_file()

    chars, char2index, index2char = create_vocabulary_dictionary(text)

    x, y, maxlen = create_syllable(text, chars, char2index)

    model = create_model(maxlen, chars)


    """ learn 1 time (1 epoch) """
    def on_epoch_end(epoch, _):
        print (f'\nEpoch: {epoch}')
        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print ('\nDiversity: ', diversity)
            generated = ''
            sentence = text[start_index : start_index + maxlen]
            generated += sentence
            print (f"Seed: {sentence}")
            sys.stdout.write(generated)
            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char2index[char]] = 1.
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = index2char[next_index]
                sentence = sentence[1:] + next_char
                sys.stdout.write(next_char)
                sys.stdout.flush()


    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    model.fit(x, y,
             batch_size=128,
             epochs=30,
             callbacks=[print_callback])


if __name__ == "__main__":

    sys.exit(main())

    