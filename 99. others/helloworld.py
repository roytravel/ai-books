import cv2
import sys
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('C:/Users/Administrator/AppData/Local/Programs/Python/Python38-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

imgFile = "C:/eol.jpg"
image = cv2.imread(imgFile)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize = (12, 8))
plt.imshow(grayImage, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()