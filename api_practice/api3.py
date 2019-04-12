import cv2
import numpy as np
import dlib
import math
# import utils
# import image_utils
# import  video_utils
# from rest_framework.response import Response
# from rest_framework.decorators import api_view

# def bilateral(im1):

def f(p, q, s): 
  return math.exp(-(abs(p-q) ** 2)/ (2 * (s ** 2)))

def g(Ip, Iq, r):
  return math.exp(-(((Ip - Iq) ** 2)/(2 * (r ** 2))))

path = "../face3.jpeg"
 
orig = cv2.imread(path)

f = cv2.bilateralFilter(orig, -1, 10, 10)
f2 = cv2.bilateralFilter(orig, -1, 80, 80)
f3 = cv2.bilateralFilter(orig, -1, 127, 127)
f4 = cv2.bilateralFilter(orig, -1, 127, 10)
f5 = cv2.bilateralFilter(orig, -1, 10, 127)
f6 = cv2.bilateralFilter(orig, -1, 80, 10) 
f7 = cv2.bilateralFilter(orig, -1, 10, 80)
f8 = cv2.bilateralFilter(orig, -1, 127, 80)
f9 = cv2.bilateralFilter(orig, -1, 80, 127)

cv2.imwrite("f.jpg", f)
cv2.imwrite("f2.jpg", f2)
cv2.imwrite("f3.jpg", f3)
cv2.imwrite("f4.jpg", f4)
cv2.imwrite("f5.jpg", f5)
cv2.imwrite("f6.jpg", f6)
cv2.imwrite("f7.jpg", f7)
cv2.imwrite("f8.jpg", f8)
cv2.imwrite("f9.jpg", f9)

