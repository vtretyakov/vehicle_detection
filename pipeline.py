import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import time
from random import randint
from scipy.ndimage.measurements import label
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
#from class_functions import *
from sklearn.model_selection import train_test_split

#from class_functions import add_heat
from class_functions import extract_features, get_hog_features
from utils import *

#image = mpimg.imread('test_images/test1.jpg')
show_plot = False

# Read in cars and notcars
#cars = glob.glob('training_data/vehicles/**/*.png', recursive=True)
#notcars = glob.glob('training_data/non-vehicles/**/*.png', recursive=True)
cars = glob.glob('training_data_subset/vehicles_smallset/**/*.jpeg', recursive=True)
notcars = glob.glob('training_data_subset/non-vehicles_smallset/**/*.jpeg', recursive=True)

# load car images
car_images = []
for image_path in cars:
    image = mpimg.imread(image_path)
    if image_path.endswith('.png'):
        image = image.astype(np.float32)/255
    car_images.append(image)

# load non-car images
notcar_images = []
for image_path in notcars:
    image = mpimg.imread(image_path)
    if image_path.endswith('.png'):
        image = image.astype(np.float32)/255
    notcar_images.append(image)

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

if show_plot == True:
    plot_images(example_images, (4, 2), fig_size=(10, 5),titles=titles)

# parameters of feature extraction
color_space = 'HLS' # Can be GRAY, RGB, HSV, LUV, HLS, YUV, YCrCb
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

if show_plot == True:
    plot_images(hog_features_examples, (6, 2), fig_size=(20, 6))

car_features = extract_features(car_images, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcar_images, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

# Create an array stack, NOTE: StandardScaler() expects np.float64
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
random_state = randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=random_state)

print('Using:',orient,'orientations',pix_per_cell,
      'pixels per cell and', cell_per_block,'cell(s) per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()


