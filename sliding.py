# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:19:46 2017

@author: Chris
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import threshholding as th
import matplotlib.image as mpimg
from scipy import signal
from PIL import Image
from moviepy.editor import VideoFileClip
import perspective as pers
import glob


def slide_windows(img):
    
 
    original_image, binary_warped, perspective_M, color_binary = th.warp_and_threshholding(img)

    
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 15
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 150
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
    
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 3) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 3) 
 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
        # Fit a second order polynomial to each
    if len(leftx) > minpix:
        left_fit  = np.polyfit(lefty, leftx, 2)
     
        if len(rightx) > minpix:    
            right_fit  = np.polyfit(righty, rightx, 2)
             
            
        else:
            print("right fit requires more points!", len(rightx) )
            
            right_fit = [0,0,0]
    else:
        print("left fit requires more points!", len(leftx))
        
        left_fit = [0,0,0]


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    original_image, top_down, perspective_Ms = pers.warp_image(out_img , True)

    
    plt.figure(0)
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='white', linewidth=2)
    plt.plot(right_fitx, ploty, color='white', linewidth=2)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.figure(1)
    plt.imshow(top_down)
    plt.figure(2)
    plt.imshow(img)
    
    return out_img, left_fit, right_fit


def scan_rows_near_fit(img, left_fit, right_fit):
   # print ("left_fit", left_fit)
    
    original_image, binary_warped, perspective_M, color_binary = th.warp_and_threshholding(img)
    
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    original_image, top_down, perspective_Ms = pers.warp_image(out_img , True)

    
    plt.figure(0)
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='white', linewidth=2)
    plt.plot(right_fitx, ploty, color='white', linewidth=2)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.figure(1)
    plt.imshow(top_down)
    
    fit_line_width = 20
        # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    fit_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    ## draw fit lines 
    left_fit_pts_l = np.array([np.transpose(np.vstack([left_fitx-fit_line_width, ploty]))])
    left_fit_pts_r = np.array([np.flipud(np.transpose(np.vstack([left_fitx+fit_line_width, 
                                  ploty])))])

    left_fit_pts = np.hstack((left_fit_pts_l, left_fit_pts_r))
    
    
    right_fit_pts_l = np.array([np.transpose(np.vstack([right_fitx-fit_line_width, ploty]))])
    right_fit_pts_r = np.array([np.flipud(np.transpose(np.vstack([right_fitx+fit_line_width, 
                                  ploty])))])

    right_fit_pts = np.hstack((right_fit_pts_l, right_fit_pts_r))
    
    cv2.fillPoly(fit_img, np.int_([left_fit_pts]), (255,255, 255))
    cv2.fillPoly(fit_img, np.int_([right_fit_pts]), (255,255, 255))
    
    result = cv2.addWeighted(out_img, 1, window_img, 0.2, 0)
    result = cv2.addWeighted(result, 1, fit_img, 0.3, 0)
    original_image, top_down, perspective_Ms = pers.warp_image(result , True)
    plt.figure(2)
    plt.imshow(top_down)

    
    return out_img
    
if __name__ == "__main__":
    
#    white_output = "C:/Users/Chris/SDC_projects/CarND-Advanced-Lane-Lines-P4/output_project_video.mp4"
#    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
#    ## To do so add .subclip(start_second,end_second) to the end of the line below
#    ## Where start_second and end_second are integer values representing the start and end of the subclip
#    ## You may also uncomment the following line for a subclip of the first 5 seconds
#    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#    clip1 = VideoFileClip("C:/Users/Chris/SDC_projects/CarND-Advanced-Lane-Lines-P4/project_video.mp4")
#    white_clip = clip1.fl_image(slide_windows) #NOTE: this function expects color images!!
#    white_clip.write_videofile(white_output, audio=False)

#     namelist = glob.glob('C:/Users/Chris/SDC_projects/CarND-Advanced-Lane-Lines-P4/test_images/test6.jpg')
#        
#     for filename in namelist:
#        img = mpimg.imread(filename)
#        print ("current image name ", filename)
#        out_img, left_fit, right_fit = slide_windows(img)
#        scan_rows_near_fit(img, left_fit, right_fit)
#        

    
    
    
    white_output = "C:/Users/Chris/SDC_projects/CarND-Advanced-Lane-Lines-P4/output_project_video.mp4"
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("C:/Users/Chris/SDC_projects/CarND-Advanced-Lane-Lines-P4/project_video.mp4").subclip(22,29)
    img = clip1.get_frame(32)
    print ("current image shape ", img.shape)
   # white_clip = clip1.fl_image(slide_windows) #NOTE: this function expects color images!!
  #  white_clip.write_videofile(white_output, audio=False)    
    
    out_img, left_fit, right_fit = slide_windows(img)
    out_img = scan_rows_near_fit(img, left_fit, right_fit)
     
    





    

