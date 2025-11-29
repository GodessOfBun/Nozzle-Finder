import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class NozzleFinder():
    def __init__(self, img):

        self.img = blur = cv.GaussianBlur(img,(5,5),0)
        self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        self.min_nozzle_size = 2390
        self.max_nozzle_size = 52795

        self.circle_tolerance = 1.4
        self.threshold_mode = 0
        self.threshold_value = 0
        # Todo: implement dynamic ROI
        

    def remove_noise(self, threshold):
        """ uses erosion and dilation to remove unwanted spots from threshold."""

        # Open and close to get rid of small spots
        kernel = np.ones((7,7),np.uint8)
        opening = cv.morphologyEx(threshold,cv.MORPH_CLOSE,kernel, iterations = 1)
        opening = cv.morphologyEx(opening,cv.MORPH_OPEN,kernel, iterations = 1) 


        # Calculate sure background area
        kernel = np.ones((5,5),np.uint8)
        sure_bg = cv.dilate(opening,kernel,iterations=3)

        # Calculate sure foreground area
        dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
        cv.imshow("trans", np.uint8(dist_transform))
        ret, sure_fg = cv.threshold(dist_transform, 20,255,0) # TODO: 20 is a magic number, 


        # Calculate unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        
        # Now, mark the region of unknown with zero
        gray = self.gray.copy()
        markers[unknown==255] = 0
        markers = cv.watershed(self.img, markers)
        gray[markers == 1] = 0
        gray[markers == 2] = 255
        gray[markers == 3] = 200
        gray[markers == 4] = 150
        gray[markers == 5] = 100
        gray[markers > 5] = 50
        
        return gray

    def find_circle(self, gray):
        
        threshold = self.get_threshold(gray)
        threshold = self.remove_noise(threshold)

        # Find contours
        contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Discard small and big contours and simplify them.
        contours = [cv.approxPolyDP(x, 0.015*cv.arcLength(x,True),True)  for x in contours 
                    if cv.contourArea(x) > self.min_nozzle_size and cv.contourArea(x) < self.max_nozzle_size]
        
        circles = []
        # Compare contour to ideal circle.
        if(contours):
            for cnt in contours:
                cir = cv.minEnclosingCircle(cnt)
                if((cir[1] * cir[1] * 3.14) / cv.contourArea(cnt) < self.circle_tolerance): # If the difference between areas is less than tolerance, contour is a circular
                    circles.append(cir)
        
        return circles
    
    def get_threshold(self, gray):

        # Use mode 1 if mode 0 is not getting results
        if(self.threshold_mode):
            ret, thresh = cv.threshold(gray,self.threshold_value,255,cv.THRESH_BINARY_INV)
            return thresh
        else:
            ret3,thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
            return thresh


    def get_nozzle_pos(self):
        # Todo: Add video and sampling/median support
        circles = self.find_circle(self.gray)
        if(circles):
            print(f'Nozzle found with method {self.threshold_mode} at pos: {circles[0][0]}, {circles[0][1]}')
        else:
            self.threshold_mode = 1
            while(not circles):
                print(f'Attempting  with method {self.threshold_mode} thresh={self.threshold_value}')
                circles = self.find_circle(self.gray)
                self.threshold_value += 5
                
                if self.threshold_value >= 250:
                    break
            
            if(circles):
                print(f'Nozzle found with method 1 thresh={self.threshold_value} at pos: {circles}')
            else:
                print("No nozzles found :(")

        # Reset to default for next read.
        self.threshold_mode = 0
        self.threshold_value = 0

        if(circles):
            return circles[0]
        else: 
            return None


if __name__ == "__main__":

    filename = "images/img1.png"
    img = cv.imread(filename)
    nozzle = NozzleFinder(img)


    center = nozzle.get_nozzle_pos()
    if(center):
        center = (np.uint16(center[0]), np.uint16(center[1]))
        print(center)
        img = cv.circle(img, center[0], center[1], (0,255,0), 2)

        filename = filename.split(".")
        filename[0] += "_Detected"
        filename = ".".join(filename)

        output = filename.split("/")[1]
        cv.imwrite(output, img)
        print(center)