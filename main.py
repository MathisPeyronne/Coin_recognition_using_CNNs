"""
This is the main program that accesses every other functions
This programs needs those files in his directory:
    - The CNN model(LeNet_model.py) and trained weights(trained_models/lenet-V2_0.9921052631578947)
    - The Cropping algorithm(cropping_algo.py)
    - A test set(X_test_64.pickle and Y_test_OH_64.pickle) 
"""

################################# Dependencies #################################

print("loading dependencies...")
import numpy as np  
import argparse
import cv2
import time
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from LeNet_model import *  # import the CNN model
import pickle
# My functions
from cropping_algo import cropping_algo  #import the cropping algorithm
import tkinter as tk            #tkinter is the library used for the file dialog
from tkinter import filedialog 
import matplotlib.pyplot as plt 

root = tk.Tk()       
root.withdraw()  # To get rid of an unecessary tkinter window


################################# Load the datasets used to compute accuracy #################################

train = {}
test = {}

# The features dataset
pickle_in = open("X_test_64.pickle","rb")
test['features'] = pickle.load(pickle_in)
test['features'] = test['features']/255.0  # Normalize the data from range 0-255 to range 0-1

# The labels dataset
pickle_in = open("Y_test_OH_64.pickle","rb")
test['labels'] = pickle.load(pickle_in)


# Variables needed later
Weights_directory = './trained_models/lenet-V2_0.9918859649122806'   
CATEGORIES = ["2.-", "1.-", "0.50.-", "0.20.-", "0.10.-", "0.05.-", "back_fr", "back_rp"]

# Start a tensorflow session
with tf.Session() as session: 
    # Restore the tensorflow session that has the model 
    saver.restore(session, Weights_directory)  
    # for i in range(10):
    #     img = test['features'][i]*255
    #     new_array = cv2.resize(img, (64, 64)).copy()
    #     plt.imshow(new_array, cmap='gray')
    #     plt.show()

    # Compute the accuracy of the current model
    test_accuracy = evaluate(test['features'], test['labels'])

    # Print the accuracy of the current model
    for i in range(30):
        print()
    print("The running model has a test accuracy of {:.3f}".format(test_accuracy))
    print()

    # Create a loop that will serve the user until he exits
    while(True):
        print('Select the image path please')

        
        try:
            # Display a file dialog
            (img_path, ) = filedialog.askopenfilenames()
        except:
            # If the user exits the file dialog:
            print('Goodbye!')
            print()
            break   # This break will exit the while(True) loop
        
        # Here we now have know the img_path that the user chose

        print("good choice !    "+img_path)
        print()

        # Apply the cropping algorithm to the image that the user chose (cf cropping_algo.py for the format of cropped_coins and bounding_circles)
        cropped_coins, bounding_circles = cropping_algo(img_path) 
        
        ################# Show the bounding boxes found by the cropping algorithm ################
        
        # Load the original image
        Color_img = cv2.imread(img_path, 1) 
        # For every coin detected
        for i in range(len(bounding_circles)):
            # Retrieve the center and the radius
            x = bounding_circles[i][0]
            y = bounding_circles[i][1]
            r = bounding_circles[i][2]
            # Draw the bounding boxe with opencv
            #cv2.rectangle(Color_img, (x-r, y-r), (x+r, y+r), (0,255,0), 10)
            cv2.circle(Color_img, (x, y), r, (0,0,255), 8)
        
        # Display the image showing the bounding boxes
        cv2.imwrite('clos_up.png', Color_img)
        cv2.namedWindow('Inputs of the Convolutional Neural Network', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Inputs of the Convolutional Neural Network', int(2976/4),int(3968/4))
        cv2.imshow('Inputs of the Convolutional Neural Network', Color_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        ########################### Feed the cropped images to the CNN ##########################
        cropped_coins = np.float32(cropped_coins)
        # Use the predict method defined in LeNet_model.py
        predictions = predict(cropped_coins)

        ################## Draw everything on the original image and display it #################
        # Load the original image
        Color_img = cv2.imread(img_path, 1)
        
        # For every coin detected
        for i in range(len(bounding_circles)):
            # Retrieve the center and the radius
            x = bounding_circles[i][0]
            y = bounding_circles[i][1]
            r = bounding_circles[i][2]
            # Draw a bounding circle
            cv2.circle(Color_img, (x, y), r, (0,255,0), 5)
            # Draw the predicted coin value
            cv2.putText(Color_img, CATEGORIES[predictions[i]], (x-50 , y+20), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0,0),5, cv2.LINE_AA)
        
        # Display image
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', int(2976/4),int(3968/4))
        cv2.imwrite('good_IRL_output2.png', Color_img)
        cv2.imshow('image', Color_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

	
