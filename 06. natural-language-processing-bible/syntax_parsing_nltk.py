# -*- coding:utf-8 -*-
"""
Title: phrase-structure syntax analysis using NLTK
Description: Create a rule for phrase-structure syntax and compile the rule using NLTK 
"""

import nltk # pip install nltk==3.3


if __name__ == '__main__':

    #Create a rule for phrase-structure syntax.

    grammar = nltk.CFG.fromstring("""
    S -> NP VP
    NP -> NN XSN JK | NN JK
    VP -> NP VP | VV EP EF
    NN -> '아이' | '케이크'
    XSN -> '들'
    JK -> '이' | '를'
    VV -> '먹'
    EP -> '었'
    EF -> '다'
    """)

    #It also offer an algorithm such as ChartParser but also ShiftReduceParser, RecursiveDescentParser. 

    parser = nltk.ChartParser(grammar)

    sent = ['아이', '들', '이', '케이크', '를', '먹', '었', '다']
    for tree in parser.parse(sent):
        print (tree)