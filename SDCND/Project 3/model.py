#!/usr/bin/env python

# References used to create this code:
#[1] - https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.y6rpzgwxc
#[2] - https://carnd-forums.udacity.com/questions/36068363/i-cannot-get-that-car-around-the-track
#[3] - https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
#[4] - https://github.com/commaai/research/blob/master/train_steering_model.py
#[5] - https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet
#[6] - https://medium.com/@tantony/training-a-neural-network-in-real-time-to-control-a-self-driving-car-9ee5654978b7#.m1scadq8f
#[7] - https://carnd-forums.udacity.com/pages/viewpage.action?pageId=32113760
#[8] - https://hackernoon.com/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a#.a1vpgjreu
#[9] - https://arxiv.org/pdf/1604.07316v1.pdf
#[10]- https://medium.com/@arnaldogunzi/teaching-a-car-to-drive-himself-e9a2966571c5#.7l4at61en

import cv2
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ELU, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from keras.optimizers import Adam

### CONSTANTS
IMAGE_WIDTH      = 200  #NVIDIA
IMAGE_HEIGHT     = 66   #NVIDIA
#IMAGE_WIDTH      = 320  #COMMAI
#IMAGE_HEIGHT     = 160   #COMMAI
EPOCHS           = 5
TRAINING_SAMPLES = 25088
BATCH            = 128
HOOD_CUT         = 20
SKY_CUT          = 40
USE_CENTER_CAMERA= 1
USE_LEFT_CAMERA  = 1
USE_RIGHT_CAMERA = 1
LEARN_RATE       = 1e-5
AUGMENT_IMGS     = 1
STEER_THRESH     = 0.1
STEER_THRESH_UP  = 100.0
THROTTLE_CMD     = 0.20
NVIDIA_DROPOUT   = 0.25
steeringangstore = []

# Function image_cuthood
### Cut car hood out of images
def image_cuthood(image,cutout):
    shape = image.shape
    image = image[0:shape[0] - cutout, 0:shape[1]]
    return image

# Function image_cutsky
### Cut car hood out of images
def image_cutsky(image,cutout):
    shape = image.shape
    image = image[cutout:shape[0], 0:shape[1]]
    return image

# Function image_flip
### flip images
def image_flip(image):
    image = cv2.flip(image,1)
    return image

# Function image_brightness
### randomly change brighness of images
def image_brightness(image):
    random_bright = .35 + np.random.uniform()
    image[:,:,0] = image[:,:,0]*random_bright
    return image

# Function image_translate
### translate images
def image_translate(image, steer, trans_range):
    rows, cols, channels = image.shape
    tr_x      = trans_range * np.random.uniform() - trans_range / 2.0
    steer_ang = steer + tr_x / trans_range * 2.0 * 0.2
    tr_y      = 0
    Trans_M   = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr  = cv2.warpAffine(image, Trans_M, (cols,rows))
    yield image_tr
    yield steer_ang

# Function image_rgb_yuv
#### Convert image from RGB to YUV and normalize Y channel
def image_rgb_yuv(img_rgb):
    img_yuv           = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    img_ych           = img_yuv[:,:, 0]
    img_max           = np.max(img_ych)
    img_min           = np.min(img_ych)
    img_ych           = (img_ych - img_min)/(img_max-img_min)*255
    img_yuv[:, :, 0]  = img_ych
    return img_yuv

## Function plot_img
### Plots an image
def plot_img(image,idx):
    import matplotlib.pyplot as plt
    plt.figure(idx)
    plt.imshow(image)
    plt.show()
    return

# Function load_csv
### Load csv
def read_csv():
    testdata = pd.read_csv('/home/honeywell/Desktop/UdacitySDCN/Project3/data/driving_log.csv')
    img_list_center = '/home/honeywell/Desktop/UdacitySDCN/Project3/data/' + testdata['center']
    img_list_left   = '/home/honeywell/Desktop/UdacitySDCN/Project3/data/' + testdata['left']
    img_list_right  = '/home/honeywell/Desktop/UdacitySDCN/Project3/data/' + testdata['right']
    print('Number of center images loaded: ', len(img_list_center))
    print('Number of left images loaded: ', len(img_list_left))
    print('Number of right images loaded: ', len(img_list_right))
    print('Number of steering commands loaded: ', len(testdata['steering']))
    steercmd = testdata['steering']
    X_data   = []
    y_data   = []
    if (USE_CENTER_CAMERA):
        for idx in range(len(steercmd)):
            prob = np.random.random()
            if((abs(steercmd[idx]) > STEER_THRESH) and (abs(steercmd[idx]) < STEER_THRESH_UP) or (prob >1.0)):
                X_data.append(img_list_center[idx])
                y_data.append(steercmd[idx])
        print('Added CENTER Camera Images')
    if(USE_LEFT_CAMERA):
        for idx in range(len(steercmd)):
            prob = np.random.random()
            if((abs(steercmd[idx]) > STEER_THRESH) and (abs(steercmd[idx]) < STEER_THRESH_UP) or (prob >1.0)):
                X_data.append(img_list_left[idx])
                y_data.append(steercmd[idx]+0.28)
        print('Added LEFT Camera Images')
    if(USE_RIGHT_CAMERA):#Throw away steering commands with low angles
        for idx in range(len(steercmd)):
            prob = np.random.random()
            if((abs(steercmd[idx]) > STEER_THRESH) and (abs(steercmd[idx]) < STEER_THRESH_UP) or (prob >1.0)):
                X_data.append(img_list_right[idx])
                y_data.append(steercmd[idx]-0.28)
        print('Added RIGHT Camera Images')
    print('X_data shape: ', np.shape(X_data))
    print('y_data shape: ', np.shape(y_data))
    x_train, x_validate, y_train, y_validate = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    print('x_train shape: ', np.shape(x_train))
    print('y_train shape: ', np.shape(y_train))
    print('x_validate shape: ', np.shape(x_validate))
    print('y_validate shape: ', np.shape(y_validate))
    yield np.array(x_train)
    yield np.array(y_train)
    yield np.array(x_validate)
    yield np.array(y_validate)

def process_image(img,target_size):
    img  = image_rgb_yuv(img)
    img  = image_cuthood(img,HOOD_CUT)
    img  = image_cutsky(img,SKY_CUT)
    img  = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return img

def keras_generator(batch_size, X_data, y_data,aug1=0,aug2=0,aug3=0,aug4=0):
    while 1:
        batch_X = []
        batch_y = []
        sz_Xdata = len(X_data)
        for i in range(batch_size):
            idx   = random.randint(0, sz_Xdata-1)
            img   = cv2.imread(X_data[idx])
            img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img   = process_image(img,target_size=(IMAGE_WIDTH,IMAGE_HEIGHT))
            steer = y_data[idx]
            if (AUGMENT_IMGS):
                prob = np.random.random()
                if (prob < 0.20):     #Don't change 20% of images
                    img   = img
                else :
                    prob = np.random.random()
                    if (prob > 0.6): #Translate 40% of images
                        img, steer = image_translate(img, steer, 100)
                        #aug1 = aug1 + 1
                        #print('translate',aug1)
                    prob = np.random.random()
                    if (prob > 0.5): #Flip 50% of images
                        img        = image_flip(img)
                        steer      = -1.0 * y_data[idx]
                        #aug2 = aug2 + 1
                        #print('flip',aug2)
                    prob = np.random.random()
                    if (prob > 0.6): #Randomly Darken 40% of images
                        img        = image_brightness(img)
                        #aug3 = aug3 + 1
                        #print('bright',aug3)
            batch_X.append(img)
            batch_y.append(steer)
            steeringangstore.append(steer)
        yield (np.array(batch_X),np.array(batch_y))

def train():
    nnmodel = model_vy(shape = [IMAGE_HEIGHT,IMAGE_WIDTH,3])
    x_data, y_data, x_validate, y_validate = read_csv()
    # Hyper Parameters
    VALIDATION_SAMPLES = len(x_validate)
    print("Length Validation Samples",len(x_validate))
    nnhistory = nnmodel.fit_generator(keras_generator(BATCH, x_data, y_data), samples_per_epoch=TRAINING_SAMPLES, nb_epoch=EPOCHS, verbose=1, nb_val_samples=VALIDATION_SAMPLES,validation_data=keras_generator(BATCH, x_validate, y_validate))
    print("The training loss is: %.3f:" % nnhistory.history['loss'][-1])
    print("The validation loss is: %.3f:" % nnhistory.history['val_loss'][-1])
    print(nnhistory.history)
    nnmodel.save_weights("model.h5", True)
    print("Model Weights Saved to model.h5")
    with open('model.json', 'w') as outfile:
        json.dump(nnmodel.to_json(), outfile)
    print("Model Saved to model.json")
    plot_steeringhist(steeringangstore)
    print("Total Steering Angles: ",len(steeringangstore))

def plot_steeringhist(steercmds):
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.hist(steercmds * 25, bins=100)
    plt.xlabel('Steering Angles', fontsize=18)
    plt.ylabel('Data Frequency', fontsize=18)
    plt.title('Histogram of Steering Data', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()

# VIVEK YADAV's model
def model_vy(shape):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - .5, input_shape=shape))
    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
    model.add(Convolution2D(32, 3, 3, activation='elu'))
    model.add(Convolution2D(32, 3, 3, activation='elu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, activation='elu'))
    model.add(Convolution2D(128, 3, 3, activation='elu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer="adam",metric=['accuracy'])
    return model


# COMMAI model
def model_commai(shape):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
              input_shape=shape,
              output_shape=shape))
    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", activation='elu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='elu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(loss ='mse', optimizer=Adam(lr=LEARN_RATE))
    return model

# NVIDIA model
def model_nvidia(shape):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - .5, input_shape=shape))
    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
    model.add(Convolution2D(24, 5, 5, activation='elu',subsample=(2, 2)))
    #model.add(Dropout(NVIDIA_DROPOUT))
    model.add(Convolution2D(36, 5, 5, activation='elu',subsample=(2, 2)))
    #model.add(Dropout(NVIDIA_DROPOUT))
    model.add(Convolution2D(48, 5, 5, activation='elu',subsample=(2, 2)))
    #model.add(Dropout(NVIDIA_DROPOUT))
    model.add(Convolution2D(64, 3, 3, activation='elu',subsample=(1, 1)))
    #model.add(Dropout(NVIDIA_DROPOUT))
    model.add(Convolution2D(64, 3, 3, activation='elu',subsample=(1, 1)))
    #model.add(Dropout(NVIDIA_DROPOUT))
    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dropout(NVIDIA_DROPOUT))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(NVIDIA_DROPOUT))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(NVIDIA_DROPOUT))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=Adam(lr=LEARN_RATE))
    return model

if __name__ == '__main__':
    train()