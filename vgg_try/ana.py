import os
import cv2

pth = "E:/xiaoqi/ApplePests62/JPEGImages/"

for par,dirs,files in os.walk(pth):
    for file in files:
        img = cv2.imread(pth+file,-1)
        print(img.shape)
        del img
