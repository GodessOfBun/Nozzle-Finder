import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('img1.png', cv.IMREAD_COLOR_BGR)
# img = cv.imread('MS3.jpg', cv.IMREAD_COLOR_BGR)
assert img is not None, "file could not be read, check with os.path.exists()"

# img = cv.resize(img, (1920, 1080))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# gray = cv.medianBlur(gray, 3)

# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(gray,(5,5),0)
# ret3,thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
ret, thresh = cv.threshold(gray,15,255,cv.THRESH_BINARY_INV)
 

# noise removal
kernel = np.ones((7,7),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernel, iterations = 1)
opening = cv.morphologyEx(opening,cv.MORPH_OPEN,kernel, iterations = 1) 


# sure background area
kernel = np.ones((5,5),np.uint8)
sure_bg = cv.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
cv.imshow("trans", np.uint8(dist_transform))

ret, sure_fg = cv.threshold(dist_transform, 20,255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
 
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
 
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv.watershed(img,markers)
gray[markers == 1] = 0
gray[markers == 2] = 255
gray[markers == 3] = 200
gray[markers == 4] = 150
gray[markers == 5] = 100
gray[markers > 5] = 50

# kernel = np.ones((9,9),np.uint8)
# gray = cv.erode(gray,kernel,iterations=6)
# gray = cv.dilate(gray,kernel,iterations=6)

contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# contours = [x  for x in contours if cv.contourArea(x) > 500]
contours = [cv.approxPolyDP(x, 0.015*cv.arcLength(x,True),True)  for x in contours if cv.contourArea(x) > 2390 and cv.contourArea(x) < 52795]

if(contours):
    cv.drawContours(img,contours, -1, (0,255,0), 2)
    for cnt in contours:
        (x,y),radius = cv.minEnclosingCircle(cnt)
        if((radius * radius * 3.14) / cv.contourArea(cnt) < 1.4):
            center = (int(x),int(y))
            radius = int(radius)
            cv.circle(img,center,radius,(255,0,0),2)
            print(cv.contourArea(cnt))

# circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT_ALT,1,5,
#                             param1=50,param2=0.6,minRadius=20,maxRadius=0)

# if not circles is None:
#     circles = np.uint16(np.around(circles))
#     cr_med = None
#     if len(circles[0]) > 1:
#         cr_med = np.mean(circles[0], axis=0)
#         cr_med = np.uint16(cr_med)
#         print(cr_med)
#     print(circles)
#     for i in circles[0,:]:
#         # draw the outer circle
#         cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
#         cv.circle(gray,(i[0],i[1]),i[2], 180,2)
#         # draw the center of the circle
#         cv.circle(img,(i[0],i[1]),2,(0,0,255),3)
#         cv.circle(gray,(i[0],i[1]),2, 180,3)
        
#     if(not cr_med is None):
#         cv.circle(img, (cr_med[0],cr_med[1]),cr_med[2],(255,0,255),3)
#         cv.circle(img, (cr_med[0],cr_med[1]),3,(255,0,255),3)
#         cv.circle(gray,(cr_med[0],cr_med[1]),cr_med[2], 100,3)

# cv.imshow("circles", circles)
cv.imshow("gray", gray)
cv.imshow("img", img)
cv.imshow('thresh',thresh)
cv.imshow('open',opening)
# cv.imshow('dilate',dilate)
# cv.imshow('surebg',sure_bg)
# cv.imshow('surefg',sure_fg)
# cv.imshow('unknown',unknown)

cv.waitKey(0)
cv.destroyAllWindows()