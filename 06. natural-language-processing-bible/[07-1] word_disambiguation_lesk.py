# -*- coding:utf-8 -*-
"""
Title: word disambigation using Lesk algorithm
Description: pratice on Lesk algorithm using WordNet's pre-defined information
"""

import sys
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.corpus import stopwords


#Extract the Best Sense about word between word and sentence.
def disambiguate(word, sentence, stopwords):
    """define the LEsk algorithm to get Best sense """
    best_sense = ''
    return best_sense


# Extract all tokens about definition of sense
def tokenized_gloss(sense):
    tokens = set(word_tokenize(sense.definition()))
    for example in sense.examples():
        tokens.union(set(word_tokenize(example)))
    return tokens


# Comrpison duplicated word
def compute_overlap(signature, context, stopwords):
    gloss = signature.difference(stopwords)
    return len(gloss.intersection(context))



if __name__ == "__main__":

    # Main function
    stopwords = set(stopwords.words('english'))
    sentence = ("They eat a meal")
    context = set(word_tokenize(sentence))
    word = 'eat'

    print ("Word :", word)
    syn = wordnet.synsets('eat')[1]
    print("Sense :", syn.name())
    print("Definition :", syn.definition())
    print("Sentence :", sentence)

    signature = tokenized_gloss(syn)
    print(signature)
    print(compute_overlap(signature, context, stopwords))
    print("Best Sense: ", disambiguate(word, sentence, stopwords))
