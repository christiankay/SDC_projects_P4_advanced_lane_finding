# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:56:48 2017

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




class find_lanes():
    def __init__(self):
       
        self.lanes_found = False
        self.good_lanes = False
        self.recent_left_fits = []
        self.recent_right_fits =[]
        self.fit_line_width = 20
        
        self.img = None
        self.image_height = None
        self.image_width = None
        self.nwindows = 15          # Choose the number of sliding windows
        self.margin = 150            # Set the width of the windows +/- margin
                
        self.minpix = 50             # Set minimum number of pixels found to recenter window  

        self.plotting = False        # enables plotting of images
        
 
        self.n_lanes = 10
        self.leftx = None # 
        self.lefty = None 
        self.rightx = None
        self.righty = None
        self.quality = 0.5 #defines when a lane will be added to recent lanes; 50% means that the lane score has to be at least 50% of median recent lane scores
        
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.last_fit_left = None    # last fit of left lane points
        self.last_fit_right = None   # last fit of right lane points
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.recent_lane_scores = [np.nan] * self.n_lanes
        
        self.curverad = None
        
        self.distance_from_center = None

        
        

        
        
    def test_lanes(self):
        
        
        
        print ("detected", self.detected)
        print ("radius_of_curvature", self.radius_of_curvature)
        
        
        
        
    def print_lane_stats(self):
        
        self.current_fit        
        
    def slide_windows(self, img):
        
        print ("#### searching for lane points within " , self.nwindows, " sliding windows ####")
        
     
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
        

        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base



        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one
        for window in range(self.nwindows):
        
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
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
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds] 
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        if len(self.leftx) > self.minpix:
            left_fit  = np.polyfit(self.lefty, self.leftx, 2)
            self.last_fit_left = left_fit
            if len(self.rightx) > self.minpix:    
                right_fit  = np.polyfit(self.righty, self.rightx, 2)
                self.last_fit_right = right_fit 
                self.detected = True
            else:
                print("right fit reqires more points!", len(self.rightx) )
                self.detected = False
                right_fit = np.nanmean(self.recent_right_fits, axis=0)
        else:
            print("left fit reqires more points!", len(self.leftx))
            self.detected = False
            left_fit = np.nanmean(self.recent_left_fits, axis=0)
        

            
    
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
        right_fitx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]
        
        out_img[self.lefty, self.leftx] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
        original_image, top_down, perspective_Ms = pers.warp_image(out_img , True)
    
        if self.plotting is True:
            plt.figure(0)
            plt.imshow(out_img)
            plt.plot(left_fitx, self.ploty, color='white', linewidth=2)
            plt.plot(right_fitx, self.ploty, color='white', linewidth=2)
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.figure(1)
            plt.imshow(top_down)
            plt.figure(2)
            plt.imshow(img)
            
            return out_img
    
    
    def scan_rows_near_fit(self, img):
        print (self.ploty.shape , self.leftx.shape)
        print ("#### scanning rows close to fit with margin of ", self.margin, "####")
        #print ("left_fit", self.left_fit)

        
        original_image, binary_warped, perspective_M, color_binary = th.warp_and_threshholding(img)
        
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        last_fit_left = self.last_fit_left
        last_fit_right = self.last_fit_right
        
        left_lane_inds = ((nonzerox > (last_fit_left[0]*(nonzeroy**2) + last_fit_left[1]*nonzeroy + 
        last_fit_left[2] - self.margin)) & (nonzerox < (last_fit_left[0]*(nonzeroy**2) + 
        last_fit_left[1]*nonzeroy + last_fit_left[2] + self.margin))) 
        
        right_lane_inds = ((nonzerox > (last_fit_right[0]*(nonzeroy**2) + last_fit_right[1]*nonzeroy + 
        last_fit_right[2] - self.margin)) & (nonzerox < (last_fit_right[0]*(nonzeroy**2) + 
        last_fit_right[1]*nonzeroy + last_fit_right[2] + self.margin)))  
        
        # Again, extract left and right line pixel positions
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds] 
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]
        
        # Fit a second order polynomial to each
        if len(self.leftx) > self.minpix:
            left_fit  = np.polyfit(self.lefty, self.leftx, 2)
            self.last_fit_left = left_fit
            if len(self.rightx) > self.minpix:    
                right_fit  = np.polyfit(self.righty, self.rightx, 2)
                self.last_fit_right = right_fit 
                self.detected = True
            else:
                print("right fit reqires more points!", len(self.rightx) )
                self.detected = False
                right_fit = np.nanmean(self.recent_right_fits, axis=0)
        else:
            print("left fit reqires more points!", len(self.leftx))
            self.detected = False
            left_fit = np.nanmean(self.recent_left_fits, axis=0)
            
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
        right_fitx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]
        
        
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        
        out_img[self.lefty, self.leftx] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
        original_image, top_down, perspective_Ms = pers.warp_image(out_img , True)
        
        
        if self.plotting is True:
    
        
            plt.figure(0)
            plt.imshow(out_img)
            plt.plot(left_fitx, self.ploty, color='white', linewidth=2)
            plt.plot(right_fitx, self.ploty, color='white', linewidth=2)
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.figure(1)
            plt.imshow(top_down)
            
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            window_img = np.zeros_like(out_img)
            fit_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, self.ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, 
                                          self.ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, self.ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, 
                                          self.ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))
            
            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            ## draw fit lines 
            left_fit_pts_l = np.array([np.transpose(np.vstack([left_fitx-self.fit_line_width, self.ploty]))])
            left_fit_pts_r = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.fit_line_width, 
                                          self.ploty])))])
        
            left_fit_pts = np.hstack((left_fit_pts_l, left_fit_pts_r))
            
            
            right_fit_pts_l = np.array([np.transpose(np.vstack([right_fitx-self.fit_line_width, self.ploty]))])
            right_fit_pts_r = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.fit_line_width, 
                                          self.ploty])))])
        
            right_fit_pts = np.hstack((right_fit_pts_l, right_fit_pts_r))
            
            cv2.fillPoly(fit_img, np.int_([left_fit_pts]), (255,255, 255))
            cv2.fillPoly(fit_img, np.int_([right_fit_pts]), (255,255, 255))
            
            result = cv2.addWeighted(out_img, 1, window_img, 0.2, 0)
            result = cv2.addWeighted(result, 1, fit_img, 0.3, 0)
            original_image, top_down, perspective_Ms = pers.warp_image(result , True)
            plt.figure(2)
            plt.imshow(top_down)
  
    
        
        return out_img 
    
    # Calculates curvature given lane polynomial fit and bottom vertical point
    def get_curvature(self, line_fit, y_eval):
        return ((1 + (2 * line_fit[0] * y_eval + line_fit[1]) ** 2) ** 1.5) \
                                 / np.absolute(2 * line_fit[0])    
    
    def calc_curvature(self):
        
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        
        left_fit = self.last_fit_left
        right_fit = self.last_fit_right
        y_eval = np.max(self.ploty)
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        print("#### left curvature radius" , left_curverad, "right curvature radius" ,right_curverad , "####")
        # Example values: 1926.74 1908.48
        
                # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.lefty*ym_per_pix, self.leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.righty*ym_per_pix, self.rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        self.radius_of_curvature = [left_curverad, right_curverad]
        print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m
        
    def fit_line(self, line_x, line_y, order_polynomial = 2):
        fit, residual, _, _, _ = np.polyfit(line_y, line_x, order_polynomial, full = True)
        return fit, residual        
        
    # Returns score based on on how much residual polynomial fit have returned
    def score_lanes_residual(self, left_lane_coordinates, right_lane_coordinates):
        _, left_residual = self.fit_line(left_lane_coordinates[1], left_lane_coordinates[0])
        _, right_residual = self.fit_line(right_lane_coordinates[1], right_lane_coordinates[0])
        
        normalized_left_residual = left_residual / len(left_lane_coordinates[0])
        normalized_right_residual = right_residual / len(right_lane_coordinates[0])
        
        # I have chosen to return sum of squares of residuals to penalize outliers more 
        return (normalized_left_residual ** 2) + (normalized_right_residual ** 2)
    
    # Returns slope and bend of each lane given list of it's coordinates
    def score_lanes_bend_and_slope(self, left_lane_coordinates, right_lane_coordinates):
        left_fit, _ = self.fit_line(left_lane_coordinates[1], left_lane_coordinates[0])
        right_fit, _ = self.fit_line(right_lane_coordinates[1], right_lane_coordinates[0])
                                                          
        return abs(left_fit[0] - right_fit[0]), abs(left_fit[1] - right_fit[1])   

    def score_lanes(self, left_lane_coordinates, right_lane_coordinates):
        # Empty lanes get bad score
        if (len(left_lane_coordinates[1]) == 0 or \
            len(left_lane_coordinates[0]) == 0 or \
            len(right_lane_coordinates[1]) == 0 or \
            len(right_lane_coordinates[0]) == 0):
            return 10**-5
        
        bend_scaler = 5*10**9
        slope_scaler = 4*10**7
        count_scaler = 10**12
        
        # First factor is how well polynomial fits the lanes
        score_residual = self.score_lanes_residual(left_lane_coordinates, right_lane_coordinates)
        # Second factor is how well matched are slopes and bends of the lane
        score_bend, score_slope = self.score_lanes_bend_and_slope(left_lane_coordinates, right_lane_coordinates)
        score_bend *= bend_scaler
        score_slope *= slope_scaler
        # Third factor is how many points are there in the lane. I assume that better lane has more pixels then noise.
        score_count = count_scaler / (len(left_lane_coordinates[0]) + len(right_lane_coordinates[0]))
        
        # Perform score fusion and obtain resulting score
        score = score_residual + score_bend + score_slope + score_count
#        print("score_residual", score_residual)
#        print("score_bend", score_bend)
#        print("score_slope", score_slope)
#        print("score_count", score_count)
        return np.float64(score)/ 10**9 
    
    # Evaluates polynomial and finds value at given point
    def get_x_for_line(self, line_fit, line_y):
        poly = np.poly1d(line_fit)
        return poly(line_y)
     
        
    # Calculates curvature and distance from center, in meters based on code from github (deniskraut)
    def get_curvature_and_distance_from_center(self, left_fit, right_fit, image_width, image_height):
        
#        print ("current left fit", left_fit)
#        print ("current right fit", right_fit)
        

        # Define conversions in x and y from pixels space to meters
        lane_width_pix = image_width * 0.57
        ym_per_pix = 23 / image_height # meters per pixel in y dimension
        xm_per_pix = 3.7 / lane_width_pix # meteres per pixel in x dimension
        
        left_lane_coordinates = [np.arange(0, image_height), self.get_x_for_line(left_fit, np.arange(0, image_height))]
        right_lane_coordinates = [np.arange(0, image_height), self.get_x_for_line(right_fit, np.arange(0, image_height))]
        
        # Find lanes fit in in meters
        left_fit_cr, _ = self.fit_line(left_lane_coordinates[1] * xm_per_pix, left_lane_coordinates[0] * ym_per_pix)
        right_fit_cr, _ = self.fit_line(right_lane_coordinates[1] * xm_per_pix, right_lane_coordinates[0] * ym_per_pix)
        
        # Calculate X coordinates for each fit
        y_vals_cr = np.arange(0, image_height * ym_per_pix)
        left_fit_cr_x = self.get_x_for_line(left_fit_cr, y_vals_cr)
        right_fit_cr_x = self.get_x_for_line(right_fit_cr, y_vals_cr)
        
        # Calculate points between the left and right lane polynomial fit
        center_cr_x = np.mean([left_fit_cr_x, right_fit_cr_x], axis=0)
        # Obtain middle of the lane polynomial fit, in meters
        center_fit_cr, _ = self.fit_line(center_cr_x, y_vals_cr)
        
        # Calculate curvature of the lane at the bottom of the image, in meters
        self.curverad = self.get_curvature(center_fit_cr, image_height * ym_per_pix)
        
        # Now calculate the distance form center
        # Calculate bottom point for each lane
        left_fitx_bottom_m = self.get_x_for_line(left_fit_cr, image_height * ym_per_pix)
        right_fitx_bottom_m = self.get_x_for_line(right_fit_cr, image_height * ym_per_pix)
        
        # Calculate image center, in meters
        center_ideal_m = image_width * xm_per_pix / 2
        # Calculate actual center of the lane, in meters
        center_actual_m = np.mean([left_fitx_bottom_m, right_fitx_bottom_m])
        
        # Calculate distance from center, in meters
        self.distance_from_center = abs(center_ideal_m - center_actual_m)
        

        
    
    def main_test(self, img):
        self.img = img
        self.image_height = img.shape[0]
        self.image_width = img.shape[1]
        
        if self.detected is False:
            self.slide_windows(img)
            
        else:    
            self.scan_rows_near_fit(img)   
            
        ##score lanes and append score to recent lane scores (score from deniskrut @ github)
        self.last_lane_score = self.score_lanes([self.leftx, self.lefty], [self.rightx, self.righty] )


        ### add lanes to recent lanes if current lane score is greater then 90% of median of recent lane scores            
        if self.last_lane_score >= self.quality * np.nanmedian(self.recent_lane_scores,axis=0) or np.nanmedian(self.recent_lane_scores, axis=0) is np.nan:
            
            print("##### lanes added to recent lanes #####")
            self.recent_lane_scores.append(self.last_lane_score)
            
            self.recent_left_fits.append(self.last_fit_left)
            self.recent_right_fits.append(self.last_fit_right)

                
            self.detected = True
        
            
        elif self.last_lane_score < self.quality * np.nanmedian(self.recent_lane_scores, axis=0):
            
            print ("lanes score less than", self.quality*100,"% of median of recent lane scores!!" , "last score:", self.last_lane_score,"median score: ", np.nanmedian(self.recent_lane_scores) )
            self.detected = False
    
        
        ## keep only last n_lanes
#        print("lenght recent left lane fits", len(self.recent_left_fits))
#        print("lenght recent right lane fits", len(self.recent_right_fits))
#        print("lenght recent lane score", np.nanmedian(self.recent_lane_scores))
#        print("last lane score", self.last_lane_score)

        if len(self.recent_left_fits) >= self.n_lanes:
            self.recent_left_fits.pop(0)
            
        if len(self.recent_right_fits) >= self.n_lanes:    
            self.recent_right_fits.pop(0)
            
        if len(self.recent_lane_scores) >= self.n_lanes:    
            self.recent_lane_scores.pop(0)
            
        #print("test2", self.recent_left_fits)     
            
        mean_recent_left_lane_fits = np.nanmean(self.recent_left_fits, axis=0)
        mean_recent_right_lane_fits = np.nanmean(self.recent_right_fits, axis=0)
        
        
        self.get_curvature_and_distance_from_center(mean_recent_left_lane_fits, mean_recent_right_lane_fits, self.image_width, self.image_height)
      
      #  self.calc_curvature()
        img_out = self.draw_lanes(mean_recent_left_lane_fits, mean_recent_right_lane_fits)
        
        plt.figure(0)
        plt.imshow(img_out)
        return img_out

    
    def draw_lanes(self, left_fit, right_fit):
        
        undistorted_image = pers.undist(self.img, calib_filename = "C:/Users/Chris/SDC_projects/CarND-Advanced-Lane-Lines-P4/saved_calibration.p")
        y_vals = np.arange(0, self.image_height)
    
        left_fitx = self.get_x_for_line(left_fit, y_vals)
        right_fitx = self.get_x_for_line(right_fit, y_vals)
    
        # Create an image to draw the lines on
        color_warp = np.zeros_like(self.img).astype(np.uint8)
    
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, y_vals]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, y_vals])))])
        pts = np.hstack((pts_left, pts_right))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        original_image, newwarp, perspective_Ms = pers.warp_image(color_warp , True)
        # Combine the result with the original image
        result = cv2.addWeighted(undistorted_image, 1, newwarp, .3, 0)
        
        # Obtain distance from center and curvature radius

#        print("test321", self.curverad, self.distance_from_center)
        # Print curvature and center offset on an image
        stats_text = 'Curvature: {0:.0f}m, Center offset: {1:.1f}m'.format(self.curverad, \
                                                                    self.distance_from_center, \
                                                                    )
        stats_text_2 = ('Current  lanes score: '+str(round(self.last_lane_score,2))+' recent lanes score: '+ str(round(np.nanmedian(self.recent_lane_scores),2)))
        
        print(stats_text)
        print(stats_text_2)
        text_offset = 670
        text_shift = 80
        font = cv2.FONT_HERSHEY_SIMPLEX
#        cv2.putText(result, stats_text, \
#                    (text_offset + text_shift, self.image_height - text_offset + text_shift), \
#                    font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(result, stats_text, (text_shift, self.image_height - text_offset), \
                    font, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
        
        
        text_offset = 640
        text_shift = 80
        
        cv2.putText(result, stats_text_2, (text_shift, self.image_height - text_offset), \
                    font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        return result
if __name__ == "__main__":
    
    find_lanes = find_lanes()
    
    white_output = "C:/Users/Chris/SDC_projects/CarND-Advanced-Lane-Lines-P4/output_project_video.mp4"
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("C:/Users/Chris/SDC_projects/CarND-Advanced-Lane-Lines-P4/project_video.mp4")#.subclip(22,29)
    img = clip1.get_frame(0)
    print ("current image shape ", img.shape)
    white_clip = clip1.fl_image(find_lanes.main_test) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


     
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
#     cap = cv2.VideoCapture('project_video.mp4')
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     find_lanes = find_lanes() 
#         
#        # Check if camera opened successfully
#     if (cap.isOpened()== False): 
#          print("Error opening video stream or file")
#        
#        # Read until video is completed
#     while(cap.isOpened()):
#          # Capture frame-by-frame
#          ret, frame = cap.read()
#          if ret == True:
#         
#            # Display the resulting frame
#            cv2.imshow('Frame',frame)
#            img = hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#            find_lanes.main_test(img)
#         
#            # Press Q on keyboard to  exit
##          if cv2.waitKey(25) & 0xFF == ord('q'):
##              break
#         
#          # Break the loop
#          else: 
#            break
#         
#        # When everything done, release the video capture object
#     cap.release()
#         
#        # Closes all the frames
#     cv2.destroyAllWindows()
    
    
#     namelist = glob.glob('C:/Users/Chris/SDC_projects/CarND-Advanced-Lane-Lines-P4/test_images/test3.jpg')
#     find_lanes = find_lanes()   
#     for filename in namelist:
#        img = mpimg.imread(filename)
#        print ("current image name ", filename)
#        
#        find_lanes.main_test(img)
 
#        
#        

    


      