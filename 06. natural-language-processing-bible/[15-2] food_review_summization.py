# -*- coding:utf-8 -*-
"""
Title
    - Food review extraction summerization using unsupervised learning
Description
    - Extraction summerization on amazone food review data opened by Kaggle
"""

import io
import re
import gc
import sys
import nltk
import string
import numpy as np
import modin.pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize
import gensim.models.keyedvectors as word2vec
import ray
ray.init()

#nltk.download('punkt')

""" strip and remove empty on sentences """
def split_sentences(reviews):
    for i in range(len(reviews)):
        sentences = sent_tokenize(reviews[i])
        for j in reversed(range(len(sentences))):
            sentences[j] = sentences[j].strip()
            """ pop if sentence is empty"""
            if sentences[j] == '':
                sentences.pop(j)
        reviews[i] = sentences

def loadEmbeddingMatrix(typeToLoad):
    """Embedding the Pre-trained glove"""
    if (typeToLoad == "glove"):
        EMBEDDING_FILE = pd.read_csv('glove.twitter.27B.25d.txt')
        EMBEDDING_FILE = 'glove.twitter.27B.25d.txt'
        embed_size = 25

    if (typeToLoad=="glove" or typeToLoad=="fasttext"):
        embeddings_index = dict()

        with open(EMBEDDING_FILE, encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]

                coefs = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = coefs # 50 dimensions
            print (f'Loaded {len({embeddings_index})} word vectors.')

    gc.collect()
    return embeddings_index # embedding_matrix

def calcaulte_sentence_embedding(wordList):
    pass

def get_sent_embedding(mylist):
    pass




def main():

    df = pd.read_csv('Reviews.csv')
    #print (df.head(3)['Text'])

    rev_list = list(df['Text'])
    split_sentences(rev_list)

    """ Insert splitted reviews to data frame. It is possible to insert a list to data frame at once as following code"""
    df['sent_tokens'] = rev_list

    """ Calculate the length of each review sentence """
    df['length_of_rv'] = df['sent_tokens'].map(lambda x: len(x))
    choice_length = 5
    df = df[df['length_of_rv']>choice_length]
    print (df.shape)

    """ Limit the maximum review vocab count upto 5K """
    list_sentences_train = df['Text']
    max_features = 5000

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    maxlen = 200
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    

if __name__ == "__main__":

    sys.exit(main())
