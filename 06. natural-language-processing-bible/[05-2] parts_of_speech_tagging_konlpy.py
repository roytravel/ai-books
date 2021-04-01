# -*- coding:utf-8 -*-
"""
Title: Parts of speech tagging using KoNLPy
Purpose: To comprison the result of POS using KoNLPy
"""


from konlpy.tag import Kkma
from konlpy.tag import Okt
from konlpy.tag import Hannanum
from konlpy.tag import Komoran
from konlpy.tag import Twitter



if __name__ == '__main__':

    kkma = Kkma()
    okt = Okt()
    komoran = Komoran()
    hannanum = Hannanum()
    twitter = Twitter()

    # Only Kkma can split the setences
    print ("kkma 문장 분리 : ", kkma.sentences("네 안녕하세요 반갑습니다."))

    # Comprison of Konlpy's library parts of speech
    print("okt 형태소 분석 : ", okt.pos(u"집에 가면 감자 좀 쪄줄래?")) #--> Ok
    print("kkma 형태소 분석 : ", kkma.pos(u"집에 가면 감자 좀 쪄줄래?"))
    print("hannanum 형태소 분석 : ", hannanum.pos(u"집에 가면 감자 좀 쪄줄래?"))
    print("komoran 형태소 분석 : ", komoran.pos(u"집에 가면 감자 좀 쪄줄래?"))
    print("twitter 형태소 분석 : ", twitter.pos(u"집에 가면 감자 좀 쪄줄래?")) # --> Ok