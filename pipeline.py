import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import time
import collections
from random import randint
from scipy.ndimage.measurements import label
from skimage.feature import hog
from class_functions import *
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

record_video = True

if record_video == True:
    show_plot = False
else:
    show_plot = True

# Read in cars and notcars
cars = glob.glob('training_data/vehicles/**/*.png', recursive=True)
notcars = glob.glob('training_data/non-vehicles/**/*.png', recursive=True)
#cars = glob.glob('training_data_subset/vehicles_smallset/**/*.jpeg', recursive=True)
#notcars = glob.glob('training_data_subset/non-vehicles_smallset/**/*.jpeg', recursive=True)

# load car images
car_images = []
for image_path in cars:
    if image_path.endswith('.png'):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = mpimg.imread(image_path)
    car_images.append(image)

# load non-car images
notcar_images = []

for image_path in notcars:
    if image_path.endswith('.png'):
        image = cv2.imread(image_path)
        augmented_image = cv2.flip(image,1)
        augmented_image2 = cv2.flip(image,0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
        augmented_image2 = cv2.cvtColor(augmented_image2, cv2.COLOR_BGR2RGB)
    else:
        image = mpimg.imread(image_path)
        augmented_image = cv2.flip(image,1)
        augmented_image2 = cv2.flip(image,0)
    notcar_images.append(image)
    notcar_images.append(augmented_image)
    notcar_images.append(augmented_image2)

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
color_space = 'GRAY' # Can be GRAY, RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 32  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

# subsample examples for extracting hog features
images_for_features = list (example_images [1:-1])
for i in range(len(images_for_features)):
    if hog_channel == 'ALL':
        channel = 0
        print("Warning! 'ALL' option for displaying is not supported")
    else:
        channel = hog_channel
    images_for_features[i] = convert_colorspace(images_for_features[i],cspace=color_space,channel=channel)


hog_features_examples = []
hog_features_examples.extend (images_for_features)

for img in images_for_features:
    features, hog_image = get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True)
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

#circular buffer
heat_buffer = collections.deque(maxlen=20)#20
accurate_bbox_list = []


def running_average(buffer, current_value):
    buffer.append(current_value)
    average = np.zeros((720, 1280)).astype(np.float)
    for i in range(len(buffer)):
        average += buffer[i]
    average = average/len(buffer)
    return average

#define sliding window search parameters
ystarts = [ 360, 380, 380]
ystops =  [ 500, 550, 670]
scales =  [ 1.0, 1.5, 2.5]
#ystarts = [ 400, 450]
#ystops =  [ 500, 656]
#scales =  [ 1.0, 2.0]

def process_image(image):

    car_boxes = find_cars(image, color_space, ystarts, ystops, scales, svc, X_scaler, hog_channel, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    #draw boxes
    #out_img = np.copy(image)
    #for c1, c2 in car_boxes:
        #cv2.rectangle(out_img, c1, c2, (0,0,255), 6)

    #plt.imshow(out_img)
    #plt.show ()
    
    #shape = image.shape[0], image.shape[1]
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    #print (image[:,:,0].shape)#(720, 1280)

    # Add heat to each box in box list
    heat = add_heat(heat,car_boxes)
    heat = running_average(heat_buffer, heat)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,5)#5 is the best so far with 20 filter length combination

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    if record_video == True:
        return draw_img
    else:
        return draw_img, heatmap





if record_video == True:
    clip = VideoFileClip("project_video.mp4")#.subclip(38,43)#.subclip(20,25)
    new_clip = clip.fl_image( process_image )
    new_clip.write_videofile("project_video_processed.mp4", audio=False)
else:
    #load test image
    images = glob.glob('test_images/*.jpg')
    #image = mpimg.imread('test_images/test3.jpg')
    for image_path in images:
        image = mpimg.imread(image_path)
        draw_img, heatmap = process_image(image)
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show ()

