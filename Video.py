import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

vc = cv.VideoCapture(1)

while(True):
    _, img = vc.read()

    img = cv.resize(img, (960, 540))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 3)

    # ret, thresh = cv.threshold(gray,50,255,cv.THRESH_BINARY_INV)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(gray,(5,5),0)
    ret3,thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

    # kernel = np.ones((5,5),np.uint8)
    # dilate = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=5)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)


    ret, sure_fg = cv.threshold(dist_transform, 0.4 ,1.0, cv.THRESH_BINARY)
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
    gray[markers == -1] = 255
    gray[markers != -1] = 255
    gray[markers == 1] = 0

    contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours = [x for x in contours if cv.contourArea(x) > 200]
    if(contours):
        cv.drawContours(img,contours, -1, (0,255,0), 1)
        for cnt in contours:
            (x,y),radius = cv.minEnclosingCircle(cnt)
            if((radius * radius * 3.14) / cv.contourArea(cnt) < 1.2):
                center = (int(x),int(y))
                radius = int(radius)
                cv.circle(img,center,radius,(100,0,255),2)

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
    cv.imshow('surebg',sure_bg)
    cv.imshow('surefg',sure_fg)
    cv.imshow('unknown',unknown)

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
vc.release()
cv.destroyAllWindows()