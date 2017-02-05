# SELF DRIVING CAR NANODEGREE PROJECT 3

## FORWARD
I would like to recognize Vivek Yadav, the Slack channel, Udacity SDCND forums and posts on medium.com for providing me
ideas for models, image processing and identifying pitfalls. Without these resources I would not have been able to complete 
the project.

## OVERALL APPROACH
My overall approach follows closely what has been described in Vivek Yadav's blog post as well as others on Medium.com and the CarND-Forums. I start with the data provided by Udacity with 24108 total images, and 8036 images per left right and center cameras. I convert each image to YUV format and then apply a normalization across the whole image. I then remove images with small steering angles (less than 0.1) and then I create via a Python generator more training images by translating, flipping and randomly darkening the original images. The model I used was that described by Vivek Yadav [1]. It has 3 sets of convolutional layers with dropouts in between followed by 4 fully connected layers [1]. I train the model with 25088 images/epoch and use 5 epochs.  

## UDACITY DATA
I used the left, right and center images provided by Udacity with the project for training. Each direction had 8036 images of size 320x160 for a total of 24108 images across 3 channels.
![alt tag](https://github.com/DFStewart/Udacity/blob/master/SDCND/Project%203/images/SteerRawHist.png)

## IMAGE PREPROCESSING
For each image of the UDACITY data I apply a preprocessing step as well as utilize a Python generator to generate new data. First I convert the image from BGR to RGB and then convert RGB to YUV. From the Traffic Sign Classifier project I found the YUV format to be the best format for a CNN to extract features. Second I normalized the Y channel of the image to improve contrast. The next step was to cut the hood of the car. By trial and error I found removing the bottom 20 pixels of the image remove the hood. By the same method I also remove the sky down to the top of the horizon. The final step of the preprocessing was to resize the images to 200x66. This was the size suggested by the NVIDIA paper [].

## GENERATING ADDITIONAL DATA
I found that the dataset provided was unbalanced and most of it was near 0 degree steering angles. This ended up biasing the training towards 0 steering angle. I ended up removing most of the images with steering angles less than 0.1. This left me with 6339 original images of which I took 10% for validation and the rest for training. For test data I just used the final test in the simulator as the test data case.

I needed more data at larger angles in order to improve the performance of the network. To do this I took the original images and generated additional data from it based on what Vivek Yadav [1] and Paul Heraty [5] describe in their posts. 

I setup a Python generator to generate each batch of 128 images. For each image in the batch in this generator there is a 20% probability to just use the original image, otherwise a new image based on the original image is formed. there is a 40% chance the image is translated, 50% change the image is flipped and 40% change the image is randomly darkened. This allows me to constantly feed the neural network new images it has not seen before. I describe each image augmentation below. Note that the ideas for all of these augmentations come from Vivek Yadav's blog [1].

### Original Image after Preprocessing
![alt tag](https://github.com/DFStewart/Udacity/blob/master/SDCND/Project%203/images/raw.png)

### Image Translation
I this augmentation I apply a random shift in the x axis of the image. This has the effect of skewing the image so the turn seems more or less severe. This allows us to get more turns of various magnitudes. Then I increase or decrease the size of the steering angle based on the shift of the image to make the steering command more or less severe depending upon the translation.
![alt tag](https://github.com/DFStewart/Udacity/blob/master/SDCND/Project%203/images/translated.png)

### Image Flip
In this augmentation I flip the image on itself so a right turn appears as a left and vice versa. I adjust the steering command in this case to be the opposite sign.
![alt tag](https://github.com/DFStewart/Udacity/blob/master/SDCND/Project%203/images/flip.png)

### Image Randomly Darked
In this augmentation I apply a random darkening across the image to represent different lighting conditions. The steering command is not adjusted in this case.
![alt tag](https://github.com/DFStewart/Udacity/blob/master/SDCND/Project%203/images/brightness.png)

## NEURAL NETWORK STRUCTURE
The first 2 networks I tried were the COMMAI and NVIDIA architectures. I tried following their model structures, but I never had much success. Adding more layers such as convolutional layers or dropouts did not seem to help either. The best I found was Vivek Yadav's model [1]. His model has one convolutional layer of 3 1x1 filters. This layer transofrms the color space of the images. This is followed by 3 sets of 2 convolutional layers, max pooling and dropouts. There is then a flattening layer followed by 3 fully connected layers. The exponential relu function was used as the activation function. The link to the picture below is from Yadav's webpage and visualizes his layout [1]: https://cdn-images-1.medium.com/max/800/1*47fIMy2fL2lc6Q1drpyYvQ.png
![alt tag](https://github.com/DFStewart/Udacity/blob/master/SDCND/Project%203/images/VivekYadavModel.png)

In addition to Vivek's model I added a Lamdba normalization layer in front of the model to help normalize the images. The optimizer I used was Adams with the loss error function used being MSE. I chose these after some trial and error with other optimizer and loss error function options. The learning rate I chose was 1e-5. Again trial and error seemed to show this to be the best value for training the model.

## TRAINING
I used Keras' fit_generator function to train the network. The training data was generated with a Python generator, this generator produces batches of 128 randomly augmented images. I trained 25088 images per epoch and did 5 epochs. I came to these parameters after some trial and error. I noticed that larger amounts of epochs would result in overfitting. 

## OTHER CHANGES
Based on a carnd-forums post[], I modified the drive.py file so that the steering angle affects the throttle command. The larger the steering angle the lower the speed The linear function I used was (throttle = -0.1*steering + 0.2). I tuned this by hand for what I felt was the best result.

## TESTING AND VALIDATION
I tested the network at 3 resolutions that my computer could handle (640x480, 800x600, 960x720). The Youtube video below shows the car successfully navigating the track in all 3 cases

## REFERENCES
[1] - https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.y6rpzgwxc
[2] - https://carnd-forums.udacity.com/questions/36068363/i-cannot-get-that-car-around-the-track
[3] - https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
[4] - https://github.com/commaai/research/blob/master/train_steering_model.py
[5] - https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet
[6] - https://medium.com/@tantony/training-a-neural-network-in-real-time-to-control-a-self-driving-car-9ee5654978b7#.m1scadq8f
[7] - https://carnd-forums.udacity.com/pages/viewpage.action?pageId=32113760
[8] - https://hackernoon.com/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a#.a1vpgjreu
[9] - https://arxiv.org/pdf/1604.07316v1.pdf
[10]- https://medium.com/@arnaldogunzi/teaching-a-car-to-drive-himself-e9a2966571c5#.7l4at61en


