import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import cv2

page_file = 'special/temp/page_1.png'
img = cv2.imread(page_file)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bwImage = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imshow('black and white image', cv2.resize(
    bwImage, None, fx=.15, fy=.15))
cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(bwImage, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
print("Number of detected contours:", len(cnts))
