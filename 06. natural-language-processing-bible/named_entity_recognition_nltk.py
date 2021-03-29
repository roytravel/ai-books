# -*- coding:utf-8 -*-
"""
Title: Named Entity Recognition using NLTK
Description: Pratice on extracting the named entity using NLTK.
"""

import nltk

nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

"""
NER Steps
1) Tokenize the input sentence
2) Analyze the splitted word as a morpheme
3) Analyze the word that splitted by morpheme as a named entity

"""


if __name__ == "__main__":

    sentence = "Prime Minister Boris Johnsoon had previously said the UK would leave by 31 October."

    # Tokenize
    tokens = nltk.word_tokenize(sentence)
    
    # Pos Tagging
    tagged = nltk.pos_tag(tokens) # It consist of (word, morpheme)

    # Entity Tagging
    entities = nltk.chunk.ne_chunk(tagged)
    print(entities)