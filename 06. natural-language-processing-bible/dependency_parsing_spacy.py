# -*- coding:utf-8 -*-
"""
Title: Dependency parsing using Spacy
Description: Perform the dependency parsing on english sentence using Spacy
"""

import spacy # pip install spacy==2.1.9
from spacy import displacy
# python -m spacy download en

if __name__ == '__main__':

    """
    perform the dependency parsing on english sentence
    spacy model process the sentence that consist of token as a document.
    each token tagged part of speech, dependency relation, named entity information, and so on.
    """

    '''
    token.text: token string
    token.dep_: type of dependency relation between token and token
    token.head: Governor token
    '''

    # English multi-task stastics model
    nlp = spacy.load('en_core_web_sm')
    doc = nlp('The fat cat sat on the mat')
    for token in doc:
        print(token.text, token.dep_, token.head.text)
    
    
    # Visualize the result of dependency parsing
    displacy.render(doc, style='dep', jupyter=True)