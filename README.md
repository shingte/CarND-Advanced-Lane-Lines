## README

### Use Computer Vision (OpenCV) to detect and mark the lane lines

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/checkboard1.png "Chessboard corners"
[image1]: ./output_images/checkboard2.png "Undistorted"
[image2]: ./test_images/straight_lines2.jpg "Road Transformed"
[image3]: ./output_images/undistorted.png "Undistorted image"
[image4]: ./output_images/transform.png "Perspective transform"
[image5]: ./output_images/binary_undistorted.png "Binary undistorted"
[image6]: ./output_images/binary_warped.png "Binary warped"
[image7]: ./output_images/histogram.png "Histogram"
[image8]: ./ref/color_fit_lines.jpg "Slide_window"
[image9]: ./output_images/slide_window.png "Slide_window"
[image10]: ./ref/curvature.jpg "curvature"
[image11]: ./output_images/straight_lines2.png "straight_lines2"
[video1]: ./output_images/project_video.mp4 "Project Video"
[video2]: ./output_images/challenge_video.mp4 "Challenge Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### The completed project and challenge videos are below with links to the repository. Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

[![Challenge Video](./output_images/challenge_video.gif)](https://youtu.be/DTCWy-7bp1A)

[![Project Video](./output_images/project_video.gif)](https://youtu.be/q9VLxnrdxQE)


#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is located in lines 56 through 159 of the file `util_cal.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

These are images of corner detection -

![alt text][image0] 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 

```py
mtx, dist = (array([[1.15777818e+03, 0.00000000e+00, 6.67113857e+02],
        [0.00000000e+00, 1.15282217e+03, 3.86124583e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
 array([[-0.24688507, -0.02373156, -0.00109831,  0.00035107, -0.00259866]]))
``` 

mtx is "camera matrix", and dist for "distortion coefficients".
I store these values in file `calibration_pickle.p`.

I applied this distortion correction to the test image using the `cv2.undistort()` function, then apply perspective transform to obtain this result: 

![alt text][image1]

The process of perspective transform to get the birds-eye view will be explained in next session.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

I read the camera calibration and distortion coefficients from file "calibration_pickle.p", then get the following images using lines 110-142 in util_cal.py.

To tell the difference of the original image and the undistorted image, I compute the Mean Square Error between the two images and show it here.


![alt text][image3]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Next step is to do the perspective transform on the undistorted image to get the top-down view, or the 'birds-eye view'.

To get a good perspective transform, we want the straight lines in the original image remains as straight in the top-down view, so I pick an image with long straight lines and select rectangles around them.
The source/destination points are defined in lines 197-199 of `util_cal.py`.

```py
src= [[581. 477.]
 [699. 477.]
 [896. 675.]
 [384. 675.]]
dst= [[384.   0.]
 [896.   0.]
 [896. 720.]
 [384. 720.]]
```

This generates transform matrix M and inverse transform matrix Minv as 

```py
M= [[-6.15990375e-01 -1.53219828e+00  1.03423384e+03]
 [-5.32907052e-15 -2.23996500e+00  1.06846331e+03]
 [-7.26415456e-18 -2.39405982e-03  1.00000000e+00]]
Minv= [[ 2.30468750e-01 -6.84027778e-01  4.92500000e+02]
 [-5.32907052e-15 -4.46435547e-01  4.77000000e+02]
 [-6.07153217e-18 -1.06879340e-03  1.00000000e+00]]
```

The codes to do the perspective transform is function `warp_image` in line 178 through line 184 of `util_cal.py`.
It uses the following 2 functions to perform the task -
- cv2.getPerspectiveTransform
- cv2.warpPerspective

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I tested a combination of color and gradient thresholds to generate a binary image. For the test images, gradient thresholds tend to generate more noise, so I focus on the color thresholds. I found several combinations that could work well for the project video, and decided to use the threshold that can filter and extract the yellow and white lines, as it works best for the project and challenge videos. (thresholding steps chosen are at lines 288 through 310 in `util_pipe.py`). 

Here's an example of my output for this step.  

![alt text][image5]

![alt text][image6]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Now that we have the warped (birds-eye view) binary image, we can identify the lane-line pixels then fit the lane-line using a 2nd order polynomial.

This is the histogram of the warped binary -

![alt text][image7]


Then I did the sliding window search, starting from the location with highest histogram counts.
The sliding window fit function is in lines 31 - 148 of `util_lane.py`.
The picture below illustrates the polynormal fit of the pixels -

![alt text][image8]


This is sliding window fit result of the warped binary above.

![alt text][image9]

Two improvements suggested in the rubrics are also implemented.
- Function `using_prev_fit` in line 154 through 174 of `util_lane.py`

  If a polynomial fit was found to be robust in the previous frame, then rather than search the entire next frame for the lines, just a window around the previous detection could be searched. 

- Function `get_processor` in line 13 through 74 of `main.py`

  Implemented a filter to smooth the lane detection over frames, meaning add each new detection to a weighted mean of the position of the lines to avoid jitter. 
  

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature at any point x of the function x = f(y) is given as follows:

![alt text][image10]

You can find the tutorial on radius of curvature [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php).

In my code, the radius of curvature is calculated ate lines 131 through 142 in `util_lane.py`
The vehicle offset position is in lines 296 through 302 in `util_lane.py`

The radius of curvature is inverse of curvature. This means a straight line actually has a radius of curvature approaching infinity.  

This is the reference of [U.S. government specifications for highway curvature](http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC).



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 290 through 352 in my code in `util_lane.py` in the function `mapping_fit_lane()`.  Here is an example of my result on the test image `straight_lines2.jpg`:

![alt text][image11]

It is the fitted lane line zone on top of the original video image, with 4 panels overlay on top.

The top panels in the order from left to right -
* Panel 1 - Road status information - Left/Right lane Radius of Curvature, and car position off lane center.
* Panel 2 - The fitted lanes on top of the road birds-eye view.
* Panel 3 - The original video image with histogram equalization.
* Panel 4 - Polynormal fit of the pixels.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my project video result](./output_images/project_video.mp4)

Here's a [link to my challenge video result](./output_images/challenge_video.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue of current approach and implementation is that lots of time is spent to find and fine tune the transform parameters, image filters and threshold. However, these paraments and hyperparameters are not general enough to handle road situation very different from the project video.

Due to time and bandwidth constraint, the `harder_challenge_video.mp4` remains as a challenge still. I would like to revisit this challenge later. 

Two approaches in mind - 
* Change or extend the current parameters/hyperparameters to better suit the country road and high contrast light condition in harder_challenge_video. It would be good to use methods can be adaptive to different situation.

* Use the deep learning approach - output the fitting result from the trained neural network, or use the semantic segmentation approach.