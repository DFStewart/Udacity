##SELF DRIVING CAR NANODEGREE PROJECT 5
---
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Cars_NotCars1.png
[image2]: ./output_images/Cars_NotCars2.png
[image3]: ./output_images/color_hist.png
[image4]: ./output_images/3Dcolor.png
[image5]: ./output_images/spatialbins.png
[image6]: ./output_images/gradientthresh.png
[image7]: ./output_images/HOGFeatures.png
[image8]: ./output_images/win_sizes.png
[image9]: ./output_images/sliding_windows.png
[image10]: ./output_images/classifierresults.png
[image11]: ./output_images/heatmaps1.png
[image12]: ./output_images/heatmaps2.png
[image13]: ./output_images/heatmaps3.png
[image14]: ./output_images/heatmaps4.png
[image15]: ./output_images/heatmaps5.png
[image16]: ./output_images/heatmaps6.png
[image17]: ./output_images/output_bboxes1.png
[image18]: ./output_images/output_bboxes2.png
[image19]: ./output_images/output_bboxes3.png
[image20]: ./output_images/output_bboxes4.png
[image21]: ./output_images/output_bboxes5.png
[image22]: ./output_images/output_bboxes6.png
[image23]: ./output_images/normalized_features.png
[video1]: ./output_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is the Writeup/README file.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

#### DATA ANALYSIS
I started by reading in all the `car` and `not-car` images. In total I have 8792 `car` images and 8968 `not-car` images. Conveniently this is an approximately balanced dataset for training. Here is an example of one of each of the `car` and `not-car` classes:

The code for this step is contained in the 2nd and 3rd code cells of the IPython notebook. 

![alt text][image1]
![alt text][image2]

My next step was to identify all the features I planned to extract.

#### COLOR BASED FEATURES
I then explored different features in different color spaces by examining color histograms, 3D plots of color space and spatial color binning. Some examples are shown below. The parameters were hand tuned via trial and error on the test images and project video.

The code for this step is contained in code cells 4-6 of the IPython notebook. 

![alt text][image3]
![alt text][image4]
![alt text][image5]

#### GRADIENT BASED FEATURES
Although I did not end up using gradient features, I did do some exploration into whethere they would be a good choice. I found empirically they did not improve my results on the video or test images so I did not end up including them. One examples is shown below.

The code for this step is contained in the 7th code cell of the IPython notebook. 

![alt text][image6]

#### HOG FEATURES

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

The code for this step is contained in the 8th code cell of the IPython notebook. 

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(4, 4)`:

![alt text][image7]

#### COMBINING FEATURES
My next step was to combine multiple feature extraction methods (HOG, spatial, color hist) and compare their results after training a classifier. I ended up creating 2 possible sets of feature extraction methods (basically with and without HOG) listed below. 
* Spatial Binning + Color Histogram
* Spatial Binning + Color Histogram + HOG

After training a classifier using each of these feature extraction methods, I found the method including HOG performed better, so I chose to use the Spatial Binning + Color Histogram + HOG combination.

#### NORMALIZING FEATURES
Before training a classifier I needed to normalize the features. To do this I used the `StandardScaler` function. An example of features before and after normalization is shown below.

The code for this step is contained in code cells 9-14 of the IPython notebook. 

![alt text][image23]

####2. Explain how you settled on your final choice of HOG parameters.

I settled on my final choice of HOG parameters by trial and error and iterating my algorithm on the project video.

There were 2 main parts I struggled with in tuning parameters, which color space to choose and the HOG parameters. LUV and YCrCb color spaces I found to work best on the video. I ended up chosing YCrCb because the classifier seemed to have less false positives on road signs in this color space. For HOG parameters I started out with the "standard" HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, but ended up finding that a combination of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(4, 4)` seemed to work better.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the `LinearSVC` function of sklearn's svm package. This seemed to work better for my system than decision trees. I trained the data by combining all of my training data, `cars` and `not-cars,` randomizing the data, splitting the data into 75% training and 25% test and then training a linear SVM. 

The code for this step is contained in code cells 17-20 of the IPython notebook. 

###Sliding Window Search
####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the sliding window search function provided in lecture. I initially found that I had insufficient windows over the cars to form strong heat maps allowing me to remove false positives. To fix this, I increased the overlap to 0.85 and also ran window searches over multiple window sizes, 64x64, 96x96 and 128x128. Because this was a huge amount of windows to search over and classify I limited the x coordinates of the window search range to be the right two lanes where the two nearest cars appear. I also limited the y coordinates of the window search range to narrower windows, because I felt smaller windows would be needed at further distances toward the horizon where the cars look smaller and larger windows would be needed closer to the camera where the cars look larger. An example of the window range is shown below for the 128x128 windows: 

![alt text][image8]

Here is an example of the output windows on a test image. Note the density of windows over the car, I found the more dense the windows on the car, the greater the heatmap strength and the better I was able to remove false positives.

![alt text][image9]

The code for this step is contained in code cells 15-16 of the IPython notebook. 

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I adjusted the penalty parameter `c` of the optimizer to 0.1, this seemed to improve the resulting test accuracy by 1%, other SVM parameters seemed to have little effect. More generally, I iterated on the training data, test images and project video and came to a final result that worked best for all by adjusting all of the feature extraction and window parameters. The classifier ended up having 98% accuracy on the test dataset. The images below show the results on a test image. Note that there are still some false positives that we will remove with heatmaps later.

![alt text][image10]

The code for this step is contained in the 8th code cell of the IPython notebook. 

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As we saw previously, after searching through multiple window sizes a number of false positive detections appeared. to remove them I used the heatmap method provided in lecture and tuned the threshold (to 10) for the video and test images.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from the test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the original images.

### Here are the test images and their corresponding heatmaps:
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]

### Here are the resulting bounding boxes drawn onto the original images:
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]
![alt text][image21]
![alt text][image22]

An additional step I performed was to setup a Vehicle Detection class to store the heatmaps from the last frame and combine it with the current frame. This was in an effort to average multiple frames together to get a more stable heatmap and remove more of the false positives.

The code for this step is contained in code cells 21-22 of the IPython notebook.

The code for the final video pipeline is contained in code cells 23-25 of the IPython notebook.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

####Issues/Problems:
I had a lot of issues getting tuning all the different parameters in such a way that all the test cases and video data worked properly. I ended up going nearly frame by frame in the video trying to identify why a particular frame failed. On top of that processing the video was taking anywhere from 30 minutes to 2.5 hours depending upon how small of search windows. 

I spent a great deal of time trying to get a centroid based tracking system to work, but I was never able to get it running in all cases. It was difficult to separate the false positives (road signs). As you can see from the final output, there are a few frames with one particular road sign at the 19 second mark that have a false positive I could not remove. I found it extremely difficult to remove that road sign without breaking other frames. To try and improve that, I could implement the centroid based tracking I describe in the robustness section below.

####Failure Modes:
Overall this pipeline is very fragile and will likely only work for this specific video. All of the parameters and thresholds are hand tuned for the test images and the project video. Any major changes to lighting, color changes in the road and other unforseen abnormalities in the lanes or road will cause major issues. I will continue to try and improve this system so that it works on the challenge videos.

####Robustness: 
To make this more robust I could continue to try and work on centroid based tracking. Centroid based tracking would work by finding the centroid of the bounding boxes on each car and then tracking them to the next frame, if the centroids for a particular box from frame to frame deviate by a pretuned amount, I could identify false positives more easily.
