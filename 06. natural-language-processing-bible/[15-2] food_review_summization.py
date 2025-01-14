# -*- coding:utf-8 -*-
"""
Title
    - Food review extraction summerization using unsupervised learning
Description
    - Extraction summerization on amazon food review data opened by Kaggle
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
from sklearn.cluster import KMeans 
from sklearn.metrics import pairwise_distances_argmin_min

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

def calcaulte_sentence_embedding(wordList, emb_index):
    emb_li = []
    for k in wordList:
        embedding_vector = emb_index.get(k)
        if embedding_vector is not None:
            if(len(embedding_vector) == 25):
                emb_li.append(list(embedding_vector))
    
    mean_arr = np.array(emb_li)
    return np.mean(mean_arr, axis=0)

def get_sent_embedding(mylist):
    """Assign the embedding on sentence using pre-defined function above and preprocess"""
    sent_emb = []
    n_sentences = len(mylist)
    for i in mylist:
        i = i.lower()
        wL = re.sub(" [^\w] ", " ", i).split()
        if (len(wL)>0):
            for k in wL:
                if(k in string.punctuation):
                    wL.remove(k)
            if(len(wL) <=2):
                continue
        else:
            print("Sentence Removed: ", i)
            continue

        res = list(calcaulte_sentence_embedding(wL))
        sent_emb.append(res)
    return np.array(sent_emb)
    




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
    
    emb_index = loadEmbeddingMatrix('glove')

    a = calcaulte_sentence_embedding(emb_index)

    how_many_summaries = 5000

    summary = [None] * how_many_summaries
    for rv in range(how_many_summaries):
        review = df['sent_tokens'].iloc[rv]
        enc_email = get_sent_embedding(review)
        if(len(enc_email)>0):
            n_cluster = int(np.ceil(len(enc_email)**0.5))
            kmeans = KMeans(n_cluster=n_cluster, random_state=0)
            kmenas = kmeans.fit(enc_email)

            avg = []
            closet = []
            for j in range(n_cluster):
                idx = np.where(kmenas.labels_ == j)[0]
                avg.append(np.mean(idx))

            closet, _ = pairwise_distances_argmin_min(kmenas.cluster_centers_, enc_email)
            ordering = sorted(range(n_cluster), key=lambda k: avg[k])
            summary[rv] = ' '.join([review[closet[idx]] for idx in ordering])
        else:
            print ("This is not a valid review")

    df_5000 = df.iloc[:5000]
    df_5000['PredictedSummary'] = summary
    df_5000[['Text', 'PredictedSummary']].to_csv('top_5000_summary.csv')
    

if __name__ == "__main__":

    sys.exit(main())
