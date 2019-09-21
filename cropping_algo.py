import cv2  # it is the library opencv
import numpy as np
import os
import time

def cropping_algo(img_path):
    """
    Input: path of the original 
    Output: cropped_coins: matrix(number of circles, 64, 64, 1) contraining the cropped images(already grayscale) 
            bounding_circles: list contraining x,y,r of each circle
    """
    
    bounding_circles = []
    cropped_coins = []
    extra_margin = 16 #in px. The cropping algorithm tends to underestimate the radius 
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #We read it as an grayscale image, and store it in img
    
    img_orig = img.copy()  # img will be altered, but at on line 42 we still need the original image, so we store a copy of img in img_orig
    height, width = img.shape[:2]
    max_length = 500 

    # resize the biggest side at 500px, meaning we reduce each edges by a factor of (max_length/height)
    reduction_factor = (max_length/height)
    img = cv2.resize(img, (int((reduction_factor)*width), max_length), interpolation = cv2.INTER_AREA) #cv2.resize(image, (new_x, new_y))

    #We apply the hough circles transform, thanks to opencv function cv2.HoughCircles
    #all_circs is a 2D array contraining all the circles(x, y, radius)
    all_circs = cv2.HoughCircles(img, 
                method = cv2.HOUGH_GRADIENT, 
                dp = 0.9, 
                minDist = 40, 
                param1=50, 
                param2=35, 
                minRadius=21, #27  pour la photo de proche #21 normal
                maxRadius=50) #80
    
    
    x=np.around(all_circs)   #we round the values describing the circles
    all_circs_rounded = np.uint16(x) #change the datatype to an unsigned integer (0 to 65535)
    
    for circle in all_circs_rounded[0,:]:  #we iterate through the circles.
        x = int(circle[0]/reduction_factor)  # x coordinate of the circle's center on the original image 
        y = int(circle[1]/reduction_factor)  # y coordinate of the circle's center on the original image
        r = int(circle[2]/reduction_factor)  # Radius of the circle on the original image
        if r+extra_margin >= 128:    #because we don't want to upsample images
            #          img_orig[      top         :      bottom      ,     left_edge     :    right_edge    ]    this keeps only the coin in the image
            crop_img = img_orig[y-(r+extra_margin):y+(r+extra_margin), x-(r+extra_margin):x+(r+extra_margin)].copy() 
            try:
                crop_img = cv2.resize(crop_img, (64, 64), interpolation = cv2.INTER_AREA)  # resize the coin to 64x64 px
                cropped_coins.append(crop_img) #append the image of the coin to the list that cropping_algo() has to return
                bounding_circles.append([x, y, r]) #append the information about the circle in a list that cropping_algo() has to return
            except Exception as e:   #an error occures when the coin is too close of the border
                print(e)
    
    
    
    #transforms cropped_coins into a matrix(number of circles, 64, 64, 1) instead of a list
    cropped_coins = np.array(cropped_coins).reshape(-1, 64, 64, 1) 
    return cropped_coins, bounding_circles 

"""
#################### Visualize results ###############################

img_path = "C:/Users/mathi/Desktop/TM/datasets/raw_dataset/20c face/IMG_20190525_170429.jpg"

cropped_coins, bounding_circles = cropping_algo(img_path)

Color_img = cv2.imread(img_path, 1)

for i in range(len(bounding_circles)):
    # draw the outer circle
    x = bounding_circles[i][0]
    y = bounding_circles[i][1]
    r = bounding_circles[i][2]
    cv2.circle(Color_img, (x, y), r, (0,0,255), 10)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', int(2976/4),int(3968/4))
cv2.imshow('image', Color_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("img.jpg", Color_img)

"""