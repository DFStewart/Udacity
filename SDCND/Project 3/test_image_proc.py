# This script tests image preprocessing

import model as mdl
import numpy as np
import random
import cv2
import math

# Test loading and plotting images
x_data, y_data, x_validate, y_validate = mdl.read_csv()
len_x    = len(x_data)
idx      = random.randint(0,len_x)
imgpath  = x_data[idx]
rawimg   = cv2.imread(imgpath)
l,w,h    = np.shape(rawimg)
rawimg   = cv2.cvtColor(rawimg, cv2.COLOR_BGR2RGB)
print(imgpath)
print('RAW: Image Len, Width, Channels:',l,w,h)
img      = mdl.process_image(rawimg,target_size=(mdl.IMAGE_WIDTH,mdl.IMAGE_HEIGHT))
mdl.plot_img(rawimg[:,:,0],0)
l,w,h    = np.shape(img)
print('PROCESSED: Image Len, Width, Channels:',l,w,h)
#img      = mdl.image_flip(img)
steer = y_data[idx] # dummy value for testing
img,steer = mdl.image_translate(img,steer, 100)
mdl.plot_img(img[:,:,0],1)
img = mdl.image_brightness(img)
mdl.plot_img(img[:,:,0],2)
img = mdl.image_flip(img)
mdl.plot_img(img[:,:,0],3)
# Test cutting top part of images
#cut_img = model.cut_image(img,50)
#model.plot_img(cut_img,1)

# Test RGB to YUV and contrast normalization
#yuv_img = model.image_rgb_yuv(cut_img)
#model.plot_img(yuv_img,2)
#l,w,h   = np.shape(yuv_img)
#print('FINAL: Image Len, Width',l,w)
#print('FINAL: Channel',h)
