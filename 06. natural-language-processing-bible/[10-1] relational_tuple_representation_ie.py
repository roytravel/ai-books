# -*- coding:utf-8 -*-
"""
Title
    - Representation of relational tuples in information extraction
Background
    - Need a structured data to extract information from massive unstructured text
"""

import sys

locs = [('고려대학교', 'In', '서울'),
('Naver', 'In', '성남'), 
('용인운전면허시험장', 'In', '용인'),
('NC 소프트', 'In', '성남'),
('삼성', 'In', '서울')]

def main():
    query = []

    # for (entity1, relation, entity2) in locs:
    #     if entity2 == "서울":
    #         query.append(entity1)
    # print (query)

    # Pythonic code
    query = [entity1 for (entity1, relation, entity2) in locs if entity2 == "서울"]
    print (query)

    


if __name__ == '__main__':
    sys.exit(main())


