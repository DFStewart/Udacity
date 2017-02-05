import numpy as np
import pandas as pd
import model

#Parse the steering information to images
testdata = pd.read_csv('/home/honeywell/Desktop/UdacitySDCN/Project3/data/driving_log.csv')

## Look at distribution of steering values
import matplotlib.pyplot as plt
plt.hist(testdata['steering'],bins=100)
plt.xlabel('Steering Angles',fontsize=18)
plt.ylabel('Data Frequency',fontsize=18)
plt.title('Histogram of Steering Data',fontsize=18)
plt.tick_params(axis='both',which='major',labelsize=18)
#plt.ylim((25,100))
plt.show()

## Test loading training data
#img_list = '/home/honeywell/Desktop/UdacitySDCN/Project3/data/' + testdata['center']
#r        = len(img_list)
#print('Number of center images: ', r)
#raw_images = np.empty([r,160,320,3])
#for idx in range(r):
#    raw_images[idx] = model.load_img(img_list[idx])
#y_data = testdata['steering']
#X_data = raw_images
#print('X_data shape: ',np.shape(X_data))
#print('y_data shape: ',np.shape(y_data))

