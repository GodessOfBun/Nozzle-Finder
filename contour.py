import numpy as np
import cv2 as cv

img = cv.imread('img6.png')
assert img is not None, "file could not be read, check with os.path.exists()"

img = cv.medianBlur(img,7)
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

thresh = cv.adaptiveThreshold(imgray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,5)

contours, hierarchy = cv.findContours(imgray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(img, contours, -1, (0,255,0), 3)

cv.imshow('image', img)
cv.imshow('thresh',thresh)

cv.waitKey(0)
cv.destroyAllWindows()