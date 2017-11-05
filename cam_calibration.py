# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:48:47 2017

@author: Chris
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle


def calibrate_and_save(location = 'C:/Users/Chris/SDC_projects/CarND-Advanced-Lane-Lines-P4/camera_cal/calibration*.jpg'):
       
    ## read a list of images using glob
    images = glob.glob(location)
    
    ## arrays to store object point and image points from all calibration images
    
    objpoints = [] # 3D points in real world space (all the same for the specific chess board the was used)
    imgpoints =[] # 2D points in image plane
    
    ## set board size
    board_col = 9
    board_row = 6
    
    # prepare object points, like (0,0,0) , (1,0,0), ... (6,8,0)
    objp = np.zeros((board_col*board_row,3), np.float32)
    objp[:,:2] = np.mgrid[0:board_col,0:board_row].T.reshape(-1,2) # x, y coordinates
    
    
    for image in images:
        ## read the calibration images
        img = mpimg.imread(image)
        #plt.imshow(img)
        ### convert to gray scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #find the chessboard corners
        ret, corners = cv2.findChessboardCorners(img_gray, (board_col,board_row), None)
    
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
             
            # draw the corners on img
            plt.figure(0)
            img_corners = cv2.drawChessboardCorners(img, (board_col,board_row), corners, ret)
            plt.imshow(img_corners)
      
            #print (image +' corners found!') 
        else:
            print (image +' no corners found!')
        
    print('Overall --' , len(imgpoints) , '-- calibration images were used!')        
        
      
    ##camera calibration based on found corners
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)  
    
    ## save camera calibration
    calibration = { "mtx": mtx, "dist": dist, "rvecs":rvecs, "tvecs":tvecs }
    filename = "saved_calibration.p"
    pickle.dump( calibration, open(filename , "wb" ) )
    print ('Camera calibration saved to ',filename )


if __name__ == "__main__":
    
    # run the calibration process
    calibrate_and_save('C:/Users/Chris/SDC_projects/CarND-Advanced-Lane-Lines-P4/camera_cal/calibration*.jpg')
    
    ## load calibration data 
    calibration = pickle.load( open( "saved_calibration.p", "rb" ) )
    print ('Calibration successfully loaded')
    mtx = calibration["mtx"]
    dist = calibration["dist"]
    
    
    ## read test image for undistortion
    img = mpimg.imread('C:/Users/Chris/SDC_projects/CarND-Advanced-Lane-Lines-P4/camera_cal/calibration1.jpg')
    plt.figure("Original image")  
    plt.imshow(img)

    
    ## undistort the image based on camera calibration
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    plt.figure(2) 
    plt.imshow(dst)

      

        