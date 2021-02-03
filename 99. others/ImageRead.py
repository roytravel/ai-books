# -*- coding:utf-8 -*-

import cv2

fileName = 'C:/Users/roytravel/Desktop/lena.jpg'

#이미지 읽기의 flag는 아래와 같이 세 가지가 존재한다

#1. cv2.IMREAD_COLOR : 이미지 파일을 Color로 읽어 들인다. 투명한 부분은 무시되며, Default 값이다.
original = cv2.imread(fileName, cv2.IMREAD_COLOR)

#2. cv2.IMREAD_GRAYSCALE : 이미지를 Grayscale로 읽어 들인다. 실제 이미지 처리시 중간단계로 많이 사용한다.
gray = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

#3. cv2.IMREAD_UNCHANGED : 이미지 파일을 alpha channel까지 포함하여 읽어들인다.
unchange = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)

'''
print (original.shape)
>>> (225, 400, 3)

이미지는 3차원 행렬로 return되며 행, 열, BGR로 이뤄져있다.
이미지의 크기 : 225 * 400

'''

#cv2.imshow(title, read_image)
cv2.imshow('Original', original)
cv2.imshow('Gray', gray)
cv2.imshow('Unchange', unchange)


#키 입력 대기 함수. 0일 경우 Key 입력까지 무한 대기. 특정 시간동안 대기하려면 milisecond 값 입력
key = cv2.waitKey(0)

if key == 27: #ESC
    cv2.imwrite('C:/Users/roytravel/Desktop/lenagray.png', gray)


elif key == (ord('s')): #save

    #이미지나 동영상의 특정 프레임 저장
    cv2.imwrite('C:/Users/roytravel/Desktop/lenagray.png', gray)

    #화면에 나타난 윈도우 종료
    cv2.destroyAllWindows()

