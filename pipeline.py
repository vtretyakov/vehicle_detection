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
from class_functions import extract_features

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

random_img_nr = randint(0, car_images_count-1)
plt.subplot(2, 2, 1)
plt.imshow(car_images[random_img_nr])
plt.title('A car image example')
plt.axis('off')
random_img_nr = randint(0, notcar_images_count-1)
plt.subplot(2, 2, 2)
plt.imshow(notcar_images[random_img_nr])
plt.title('Not a car image example')
plt.axis('off')
#plt.subplot_tool()
plt.show()



