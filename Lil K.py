import cv2
import matplotlib.pyplot as plt
import numpy as np


def display_lines(image, lines):
    for line in lines:
        x1, y1,x2, y2 = line
        col = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        cv2.line(image,(x1,y1),(x2,y2),col,2)
    #cv2.imshow('detection',image)
    plt.imshow(image)
    plt.show()
    #cv2.waitKey(0)
sigma = 1
thresh_hough = 100
min_ll = 5
max_lg = 50

image = cv2.imread(r"E:\Mails\Imogolites\Base_travail_Immo.jpg") #Fails if uses as-is due to bright background.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
cv2.waitKey(0)
#gray = ~gray
gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)
cv2.imshow("gray_blurred",gray_blurred)
cv2.waitKey(0)

gray_thresh = cv2.threshold(gray_blurred, 180, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("gray_thresh",gray_thresh)
cv2.waitKey(0)

edges = cv2.Canny(gray,50,150,apertureSize=3)
cv2.imshow("edges",edges)
cv2.waitKey(0)

lines = cv2.HoughLinesP(edges,sigma,np.pi/180, thresh_hough,min_ll,max_lg)
lines = np.squeeze(lines)

print(lines.shape)

display_lines(image, lines)
