import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import time
from random import randint
from scipy.ndimage.measurements import label
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
#from class_functions import *
from sklearn.model_selection import train_test_split

#from class_functions import add_heat
from class_functions import extract_features, get_hog_features
from utils import *

#image = mpimg.imread('test_images/test1.jpg')



# Read in cars and notcars
#cars = glob.glob('training_data/vehicles/**/*.png', recursive=True)
#notcars = glob.glob('training_data/non-vehicles/**/*.png', recursive=True)
cars = glob.glob('training_data_subset/vehicles_smallset/**/*.jpeg', recursive=True)
notcars = glob.glob('training_data_subset/non-vehicles_smallset/**/*.jpeg', recursive=True)

# load car images
car_images = []
for image_path in cars:
    car_images.append (mpimg.imread(image_path))

# load non-car images
notcar_images = []
for image_path in notcars:
    notcar_images.append (mpimg.imread(image_path))

car_images_count = len(car_images)
notcar_images_count = len(notcar_images)

print ('dataset has cars:', car_images_count)
print ('and not cars:', notcar_images_count)

# create a small subset of images for visualization
example_images = []
titles = []
for i in range(0,4):
    example_images.append(car_images[randint(0, car_images_count-1)])
    titles.append("car")
for i in range(0,4):
    example_images.append(notcar_images[randint(0, notcar_images_count-1)])
    titles.append("not car")

plot_images(example_images, (4, 2), fig_size=(10, 5),titles=titles)

# parameters of feature extraction
color_space = 'GRAY' # Can be GRAY, RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 1 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off

# subsample examples for extracting hog features
images_for_features = list (example_images [1:-1])

hog_features_examples = []
hog_features_examples.extend (images_for_features)

for img in images_for_features:
    features, hog_image = get_hog_features(convert_colorspace(img,cspace='HLS',channel=2), orient, pix_per_cell, cell_per_block, vis=True)
    hog_features_examples.append (hog_image)

plot_images(hog_features_examples, (6, 2), fig_size=(20, 6))


