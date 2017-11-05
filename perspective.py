# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:47:39 2017

@author: Chris
"""

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



          

def undist(img, calib_filename):
    ## load calibration data 
    calibration = pickle.load( open( calib_filename, "rb" ) )
    
    mtx = calibration["mtx"]
    dist = calibration["dist"]
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Source coordinates
  #  print ('image successfully undistorted!')
    return undist

def transform(img, calib_filename, src_points , dst_points):

    # image size
    img_size = img.shape

    # Use the OpenCV undistort() function to remove distortion

    # Source coordinates
    src = np.float32([
                    src_points["p_tl"],
                    src_points["p_tr"],
                    src_points["p_bl"],
                    src_points["p_br"],
                                        ])    
    # Search for corners in the grayscaled image
    
   
   
    # Destination coordinates
    dst = np.float32([
                    dst_points["p_tl"],
                    dst_points["p_tr"],
                    dst_points["p_bl"],
                    dst_points["p_br"],
        ])
       
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, (img_size[1],img_size[0]))
#    cv2.polylines(warped,vertices_dst,True,color=(255,0,0), thickness=4)
#    cv2.polylines(img,vertices_src,True,color=(255,0,0), thickness=4)
    return img, warped, M

def warp_image(img, reverse = False):
    calib_filename = "C:/Users/Chris/SDC_projects/CarND-Advanced-Lane-Lines-P4/saved_calibration.p"
    img_size = img.shape
    image_width = img_size[1]
    image_height = img_size[0]
    offset = 0
    ### point to define source geometry
    src_points = {"p_tl" : [image_width * 0.4475-offset, image_height * 0.642], # position top left
                 "p_tr" : [image_width * 0.5525+offset, image_height * 0.642], # position top right
            
                 "p_bl" : [image_width * 0.175-offset, image_height * 0.95], # position bottom left   
                 "p_br" : [image_width * 0.825+offset, image_height * 0.95]}  # position bottom right
                 
    ### point to define destination geometry            
    dst_points = {"p_tl" : [image_width * 0.2, image_height * 0.025], # position top left
                    "p_tr" : [image_width * 0.8, image_height * 0.025], # position top right
            
                    "p_bl" : [image_width * 0.2, image_height * 0.975], # position bottom left   
                    "p_br" : [image_width * 0.8, image_height * 0.975]}  # position bottom right
    ## masking roi for testing
#    roi = region_of_interest(img, src_points)
#    plt.imshow(roi)
    ##### use undist an dtransform and plot results
    if reverse is False:
        img = undist(img, calib_filename)
        original_image, top_down, perspective_M = transform(img, calib_filename, src_points, dst_points)
    else:
        original_image, top_down, perspective_M = transform(img, calib_filename, dst_points, src_points)
        
    
    return original_image, top_down, perspective_M 
    
if __name__ == "__main__":   
    filename = 'C:/Users/Chris/SDC_projects/CarND-Advanced-Lane-Lines-P4/test_images/straight_lines1.jpg'
    calib_filename = "C:/Users/Chris/SDC_projects/CarND-Advanced-Lane-Lines-P4/saved_calibration.p"
    img = mpimg.imread(filename)

    ##### use warp_image and transform and plot results
    original_image, top_down, perspective_M = warp_image(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
    f.tight_layout()
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=14)
    ax2.imshow(top_down)
    ax2.set_title('Undistorted and Warped Image', fontsize=14)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
