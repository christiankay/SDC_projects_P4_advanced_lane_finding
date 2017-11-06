# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:41:04 2017

@author: Chris
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import perspective as pers


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else: gray = img     
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else: gray = img 
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else: gray = img 
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output




def gradient_detection(img):
    
    

        
    #load image 
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]


    
#    hls = cv2.cvtColor(limage, cv2.COLOR_RGB2HLS)
#    channel = hls[:,:,hls_channel]
   
    ## apply blurring to improve edge detection
    image = cv2.blur(s_channel,(3,3))
    
    
    #################################################
    #### Apply each of the thresholding functions####
    #################################################
    # Choose a Sobel kernel size
    ksize_a = 3 # Choose a larger odd number to smooth gradient measurements
    ksize_b = 7 # Choose a larger odd number to smooth gradient measurements
    
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize_a, thresh=(30, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize_a, thresh=(30, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize_b, mag_thresh=(40, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize_b, thresh=(50*np.pi/2/90, 65*np.pi/2/90))
    ## opening to remove distributed random dir edges
    dir_binary= cv2.morphologyEx(dir_binary, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    ## close to establish solid lines
    dir_binary= cv2.morphologyEx(dir_binary, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    
    ## first combination of thresholding functions 
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1))  | ((mag_binary == 1) & (dir_binary == 1))] = 1
    ## close to establish solid lines
    combined =  cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    ## opening to remove distributed random dir edges
    grad_combined =  cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    
    #################################################
    #### Apply each of the thresholding functions####
    ################################################# 
    ####set img to l channel
    
    l_channel = hls[:,:,1]
   
    image = cv2.blur(l_channel,(3,3))    
    ## find grad in l channe image
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize_a, thresh=(30, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize_a, thresh=(30, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize_b, mag_thresh=(40, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize_b, thresh=(50*np.pi/2/90, 65*np.pi/2/90))
    ## opening to remove distributed random dir edges
    dir_binary= cv2.morphologyEx(dir_binary, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    ## close to establish solid lines
    dir_binary= cv2.morphologyEx(dir_binary, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    
    ## first combination of thresholding functions 
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1))  | ((mag_binary == 1) & (dir_binary == 1))] = 1
    ## close to establish solid lines
    combined =  cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    ## opening to remove distributed random dir edges
    grad_combined2 =  cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))   

    
    return img, gradx, grady, mag_binary, dir_binary, grad_combined, grad_combined2

def color_detection(img):
    
    
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,0]
    
    
    # Threshold s color channel
    s_thresh_min = 90
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    # Threshold  h color channel
    h_thresh_min = 5
    h_thresh_max = 110
    h_binary = np.zeros_like(l_channel)
    h_binary[(l_channel >= h_thresh_min) & (l_channel <= h_thresh_max)] = 1
    

    
    # Combine the two binary thresholds
    combined_color_binary = np.zeros_like(s_binary)
    combined_color_binary[(s_binary == 1) & (h_binary == 1)] = 1
    
    return combined_color_binary
#######################################
    


# Retuns binary image based on thresholding functions
def get_binary_image(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    S = hls[:,:,2]
    L = hls[:,:,1]
    
    light_mask = np.zeros_like(L)
    light_mask[(S >= 5) & (L >= 130)] = 1

    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx_l = abs_sobel_thresh(L, orient='x', sobel_kernel=ksize, thresh=(25, 100))
    gradx_s = abs_sobel_thresh(S, orient='x', sobel_kernel=ksize, thresh=(10, 100))
    
    combined = np.zeros_like(gradx_s)
    # For some images S channel works better, while for others L channel does
    # Mix results from either channels
    combined[((gradx_l == 1) | (gradx_s == 1)) & (light_mask == 1)] = 1
    
    return combined

def warp_and_threshholding(img):
    
        
        
        bin_img = get_binary_image(img)
        original_image, top_down_grad_combinded, perspective_M = pers.warp_image(bin_img)
        
        original_image, top_down_img, perspective_M = pers.warp_image(img)
       # combined_color_binary = color_detection(top_down_img)
        ## combine gradient and color thresholding
        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        
        color_binary = 0#np.dstack(( np.zeros_like(top_down_grad_combinded), combined_color_binary, top_down_grad_combinded)) 
#        
#        # Combine the two binary thresholds
#        combined_binary = np.zeros_like(bin_img)
#        combined_binary[(combined_color_binary == 1) | (top_down_grad_combinded_S == 1) ] = 1
        
#        original_image, top_down, perspective_M = pers.warp_image(combined_binary)
#        top_down[(top_down > 0)] = 1
#        top_down = np.asarray(top_down, dtype=np.uint8)
        
        return  original_image, top_down_grad_combinded, perspective_M, color_binary
    
    
if __name__ == "__main__":
    
    namelist = glob.glob('test_images/*.jpg')
    
    for filename in namelist:
        img = mpimg.imread(filename)
        print ("current image name ", filename)
        
        
        
        image, gradx, grady,mag_binary, dir_binary, grad_combined_S, grad_combined_L = gradient_detection(img)
        
        original_image, combined_binary, perspective_M, color_binary = warp_and_threshholding(img)
    

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('Stacked thresholds')
        ax1.imshow(color_binary, cmap='gray')
        
        ax2.set_title('Combined color and gradient thresholds')
        ax2.imshow(combined_binary, cmap='gray')

        
        #################################################
        ## setup font for plot titles####################
        #################################################
        font = {'family': 'arial',
                'color':  'black',
                'weight': 'normal',
                'size': 12,
                }
        ### binary gradient images#########
        plt.figure(filename+str(0))
        plt.imshow(img)     
        plt.title('Original image after blurring applied '+filename, fontdict=font)
#        plt.figure(filename+str(1))
#        plt.imshow(gradx)
#        plt.title('Gradient in x direction '+filename, fontdict=font)
#        plt.figure(filename+str(2))
#        plt.imshow(grady)
#        plt.title('Gradient in y direction '+filename, fontdict=font)
#        plt.figure(filename+str(3))
#        plt.imshow(mag_binary)
#        plt.title('Magnitude of the gradient '+filename, fontdict=font)
#        plt.figure(filename+str(4))
#        plt.imshow(dir_binary)
#        plt.title('Opening and Closing of direction of gradient '+filename, fontdict=font)
        plt.figure(filename+str(5))
        plt.imshow(grad_combined_S)
        plt.title('Combination 1 - S- channel - gradx & grady or mag_grad & dir_of_grad'+filename, fontdict=font)
        plt.figure(filename+str(6))
        plt.imshow(grad_combined_L)
        plt.title('Combination 2 - L- channel - gradx & grady or mag_grad & dir_of_grad'+filename, fontdict=font)
        #############
        #############
        ## binary color threshhold images
        
        
        
        