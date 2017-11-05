

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted.png "Originial and undistorted camera image"
[image2]: ./output_images/undistorted_right_road.png "Road Transformed"
[image3]: ./output_images/original_gradient.png "Original image for gradient detection"
[image32]: ./output_images/combination1_gradient.png "Combination (1) of color and gradient detection"
[image33]: ./output_images/combination2_gradient.png "Combination (2) of color and gradient detection"
[image34]: ./output_images/combined_color_and_gradient_threshholds.png "Stacked combination of color and gradient threshholding & final warped binary"
[image4]: ./output_images/perspective.png "Warp Example"
[image5]: ./output_images/sliding_windows.png "Sliding windows and historam evaloation to find first lane lines"
[image6]: ./output_images/search_near_fit.png "Searching lane points near last fit"
[image7]: ./output_images/curvature.png "Formular for the curvature"
[image8]: ./output_images/lane_finding_example.png "Final output of the pipeline"
[video1]: ./out_project_video_n10_q50.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 16 through 68 of the file called `cam_calibration.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient threshholds to generate binary images (threshholding steps at lines 16 through 154 in `threshholding.py`). I also used morphology based opening and closing operators ('gradient_detection()' in 'threshholds.py' to obtain binary images with potentially less noise. Here's an example of my output for this step.  

![alt text][image3]
![alt text][image32]
![alt text][image33]
![alt text][image34]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()` which appears in lines 64 through 94 in file `perspective.py`. The `warp_image()` function takes as inputs an image (`img`) and the direction of warping (reverse = True/False). Source (`src`) and destination (`dst`) points are hardcoded in this function. The points were used as arguments for a sub-function 'transform()' which finally warps the image to the destination points. I chose the source and destination points in the following manner:

```python
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
```


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used a second order polynomial to fit left and right lane lines seperately. To find sufficiant fit points, the pipeline explained above was used to get a warped binary image of possible lane line points. Based on that binary histogram evaluation and sliding windows were used to determine which points in the binary image belong to possible lane lines. (file 'lane_finder', line 89 to 208)

![alt text][image5]

Once the lane were found and a sufficiant fit (fit score evaluation can be found in line 373 to 417) was applied, the function 'scan_rows_near_fit()' in 'lane_finder.py' was called to only search for new points in the area close to the last fit (margin = 150 pixels). This area is marked (green) in the following image while the last fits (left and right) are represented by white lines: 

![alt text][image6]

In line 473 through 535 (method 'main_test()') the pipeline decision making is coded. First a if statement decides whether 'sliding_windows() or "searching for lane point near last fit" is used. For every fit pair a score based on 1. residuals 2. slope and bend and 3. number of lane points used for the fit is calculated. A lane fit will only be added to the list of last 10 lane fits if the current score is not less than 50% of the median score of last 10 lane fits. Curvature, center offset and lane drawing is baesed on the mean of 10 last sufficiant lane line fits. 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 426 through 468 in my code in `lane_finder.py`. The curvature is calculated by finding the relation of pixel to real world distances and calculating a "center" lane line based on the left and right line fits and using the following formular (course material):

![alt text][image7]

The center was found by assuming the camera to be mounted at the center of the front window and by calculating the mean of left and right fit values at the bottom of every image.



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 538 through 589 in my code in `lane_finder.py` in the function `draw_lanes()`.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_project_video_5scores_q50.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Backbone of the pipeline is threshholding and binary image creation (feature extraction) to use them as points for polynomial fits. In case of changes in the illumination the pipeline is likely to fail since the color and gradient threshholds are static and new gradients and colors might appear in the images. More robust techniques for feature extraction are neccassary like adaptive threshholding or deep neural networks.
Also information regarding the direction of the movement (optical flow) could be used to estimate the movement of the car.




