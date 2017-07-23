import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_images (images, table_size, fig_size = (10, 10), cmap=None, titles=None, fontsize=16):
    """Shows images in a table
        Args:
        images (list): list of input images
        table_size (tuple): (cols count, rows count)
        fig_size (tuple): picture (size x, size y) in inches
        cmap (list): list of cmap parameters for each image
        titles (list): list of images titles
        fontsize (int): size of the font
        """
    sizex = table_size [0]
    sizey = table_size [1]
    fig, imtable = plt.subplots (sizey, sizex, figsize = fig_size, squeeze=False)
    for j in range (sizey):
        for i in range (sizex):
            im_idx = i + j*sizex
            if (isinstance(cmap, (list, tuple))):
                imtable [j][i].imshow (images[im_idx], cmap=cmap[i])
            else:
                im = images[im_idx]
                if len(im.shape) == 3:
                    imtable [j][i].imshow (im)
                else:
                    imtable [j][i].imshow (im, cmap='gray')
            imtable [j][i].axis('off')
            if not titles is None:
                imtable [j][i].set_title (titles [im_idx], fontsize=fontsize)

    plt.show ()



def convert_colorspace(image, cspace='RGB', channel='ALL'):
    """Converts image to a selected color space
        Args:
        image: input images
        cspace (string): color space
        channel (int or string): 'ALL' channels or channel number
        """
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            conv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            conv_img = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            conv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            conv_img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            conv_img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        elif cspace == 'GRAY':
            conv_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else: conv_img = np.copy(image)
    if channel == 'ALL' or cspace=='GRAY':
        return conv_img
    else:
        return conv_img [:,:,channel]
