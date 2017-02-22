##SELF DRIVING CAR NANODEGREE PROJECT 4
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

[image1]: ./writeup_images/cam_undist.png "Camera Undistorted"
[image2]: ./writeup_images/img_undist.png "Image Undistorted"
[image3]: ./writeup_images/clr_thresh.png "Color Threshold 1"
[image4]: ./writeup_images/clr_thresh2.png "Color Threshold 2"
[image5]: ./writeup_images/HSV.png "HSV"
[image6]: ./writeup_images/HLS.png "HLS"
[image7]: ./writeup_images/RGB.png "RGB"
[image8]: ./writeup_images/sobel_test.png "Sobel Test"
[image9]:  ./writeup_images/img_prep.png1 "Image Processed 1"
[image10]: ./writeup_images/mask.png "Masked Image"
[image11]: ./writeup_images/img_prep.png "Image Processed 1"
[image12]: ./writeup_images/perspective_tfm.png "Birds Eye Transform"
[image13]: ./writeup_images/lane_det.png "Lane Find"
[image14]: ./writeup_images/nxt_frame.png "Next Frame Lane Find"
[image15]: ./writeup_images/output.png "Projected Images"

[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is the Writeup/README  file.

Note that I provided 2 IPython notebooks, one with most of the code executed so that important outputs are displayed and another with just the code unevaluated. This was due to the size of the notebook with all the code executed and output images displayed. It was over 25 MB so I had difficulty uploading and viewing it on GitHub.

* P4.ipynb - this is the code with most of the cells executed, it is very large
* P4_justcode.ipynb - this is just the code, it is very small

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first 3 code cells of the IPython notebook located in "./P4.ipynb". There is a subheading of "CAMERA CALIBRATION" that labels this part of the code.

I used the standard camera calibration pipeline described below and in the lectures.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I updated the chessboard parameters to incorporate the 9x6 squares chessboard. There are 20 total provided camera calibration images I used in the camera calibration.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (Single Images)

####1. Provide an example of a distortion-corrected image.
Using the camera calibration computed from the 20 camera calibration images I used the `cv2.undistort()` function to obtain the distortion corrected images on the test images provided with the project. This code is performed in cells 5 and 6 in the IPython notebook. An example is shown below:
![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

#### COLOR THRESHOLDING
My first step was to try and choose color thresholds from the RGB images. The lanes are white and yellow, so I tuned my color thresholds to try and identify white and yellow pixels in the image without getting too much noise from similar colors in the image. The images with the lighter grey bridge surface were most difficult because there was the least contrast between the light grey road surface and the yellow and white lane lines. I hand tuned these parameters to get the best result. This code is 
contained in the 8th cell under the COLOR THRESHOLDING TESTS subheading. The images below are a sample of the color thresholded binary image (the perspective transform is described below, but was executed before this code):
![alt text][image3]
![alt text][image4]

#### COLOR SPACE ANALYSIS
My second step was to identify which color space would work best for the gradient thresholding on. I looked at HSV, HLS and RGB (see example images below). By inspection it was clear the S-Channel of the HLS had the most prominent contrast in the lanes and I felt it was the best candidate to run the gradient thresholding on. The code for this is contained in cells 9-11 under the COLOR SPACES TESTING subheading.
* HSV
![alt text][image5]
* HLS
![alt text][image6]
* RGB
![alt text][image7]

#### GRADIENT THRESHOLDING ANALYSIS
My next step was to test the different types of Sobel gradient thresholding on the S-channel of HLS converted images and combine that with the color thresholding. I tried absolute x, absolute y, magnitude and directional Sobel gradient thresholding. As shown in the figure below, the absolute x Sobel gradient thresholding worked the best on the test images. This is contained in cell 12. I used the `cv2.Sobel()` function to compute the x gradients and then computed the magnitude in the x direction. I used a kernel size of 21 and min and max thresholds of 20 and 100 respectively. This is contained in the `sobel_absx()` function.

![alt text][image8]

#### COMBINING COLOR AND GRADIENT THRESHOLDING
To combine the color and gradient thresholds I performed a bitwise & on the binary images formed from the color and gradient thresholding. The code for this is contained in cell 13 in the functions `pipeline_binary_img_test()`, `combined_color()` and `sobel_absx()`. Here's an example of my output for this step.

![alt text][image9]

#### MASKING
I noticed in testing the video that I was getting a lot of extra lines from the sides of the bridges and edges of some of the roads. I also noticed by inspection that the lane lines mostly stayed in the birds eye projected area between x of 200 and 1200. I decided to mask out this region from the binary image after its projection to the birds eye view. This is contained in cell 14 and in the function 'region_of_interest'.
The mask verticies I used are:

| Verticies     | 
|:-------------:|
| 200, 0        |
| 200, 720      |
| 1200, 720     |
| 1200, 0       |

![alt text][image10]

#### FINAL PROCESSED OUTPUT
This is the final output binary image after color and gradient thresholding and masking.
![alt text][image11]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `BirdsEyeTransform()`, which appears in the 7th code cell of the IPython notebook.  The `BirdsEyeTransform()` function takes as inputs an image (`img`) and has hard coded values for source (`src`) and destination (`dst`) points. I chose the hardcode the source and destination points in the following manner:

```
    src = np.float32([[583, 460], [203, 720], [1127, 720], [705, 460]])
    dst = np.float32([[320, 0], [320, 720], [960,720], [960, 0]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 583, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127,720      | 960, 720      |
| 705, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image12]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the code provided in lecture to compute the lane-line pixels and fit them to a polynomial. A histogram is formed across the lower part of the image columns, the peak of which sets the center of a window that is slid up the image to follow and detect the lines. I adjusted the margin and min pixel to 110 and 100 respectively based on many iterations of running the code on the video. The code is contained in the `lane_find()` function. This is under the LANE DETECTION sub heading. Shown below is the result for the test images:

![alt text][image13]

To limit the need to blind search the next frame of video, we implement the code provided in lecture to just search a margin around the previously detected line position. This is contained in the `nextframe_lanefind()` function. Shown below is the result for the test images:

![alt text][image14]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In the COMPUTE LANE CURVATURE AND ERROR sub heading I compute the radius of curvature and lane position with respect to center. The basic code I use to compute the lane curvature is in the `ComputeLaneCurvature()` function. This is nearly identical to what was given in lecture. To compute the position of the vehicle with respect to center I compute the center of the lane curvature from the left and right lane fits and then compute the error. This code is contained in the `ComputeLanePosErr()` function.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `projectlane()`.  Here are the results on the test images:

![alt text][image15]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

My final pipeline is contained withing the `process_video()` function.

Here's a [link to my video result](./output_videos/project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

####Issues/Problems:
I had a lot of issues getting tuning all the different parameters in such a way that all the test cases and video data worked properly. I ended up going nearly frame by frame in the video trying to identify why a particular frame failed. I realized that it was going to be extremely difficult to satisfy all the frames and instead I needed to average the lane line fits across multiple frames to prevent a single bad frame from failing. 

####Failure Modes:
Overall this pipeline is very fragile. All of the parameters and thresholds are hand tuned for the test images and the project video. Any major changes to lighting, color changes in the road, changes to lines in the road and other unforseen abnormalities in the lanes or road will cause major issues.

####Robustness: 
To make this more robust I could implement more logic to try and compare the polynomials between frames to protect against cases when the lane finder failed. Also I could spend more time trying to further tune Sobel and color thresholds.
