# -*- coding:utf-8 -*-
"""
Title
    - Practice on relation extraction what applied regular expression
Description
    - General methodology to relation etraction is to find tuple form like this -> (e1, rel, e2)
"""

import re
import sys
import nltk
nltk.download('ieer')

def main():
    # (?!\b.+ing\b) is used when searcing the string that contains the word.
    IN = re.compile(r'.*\bin\b(?!\b.+ing) ')
    
    for doc in nltk.corpus.ieer.parsed_docs("NYT_19980315"):
        print ("doc : ", doc)
        for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern = IN):
            print (nltk.sem.rtuple(rel))

if __name__ == '__main__':
    sys.exit(main())


