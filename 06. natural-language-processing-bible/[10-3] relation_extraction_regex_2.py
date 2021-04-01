# -*- coding:utf-8 -*-
"""
Title
    - Practice on relation extraction what applied regular expression 2
Description
    - conll2002 Dutch corpus contain not only named entity but also pos tagging.
    this allows us to devise patterns that are sensitive to these tags.
"""

import re
import sys
import nltk
nltk.download('conll2002')

# Add a multiple regular expression

vnv = """
    (
        is/V|
        was/V|
        werd/V|
        wordt/V
    )
    .*
    van/Prep
"""


def main():
    VAN = re.compile(vnv, re.VERBOSE)

    # relation extraction about conll2002 corpus.
    for doc in nltk.corpus.conll2002.chunked_sents('ned.train'):
        for rel in nltk.sem.extract_rels("PER", "ORG", doc, corpus='conll2002', pattern=VAN):
            print("", nltk.sem.clause(rel, relsym="VAN"))

if __name__ == '__main__':
    sys.exit(main())


