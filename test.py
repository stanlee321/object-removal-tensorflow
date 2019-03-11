from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import skimage.morphology

from model import DeepModel
import skimage.morphology
import skimage.io as io
from skimage.color import rgba2rgb

#from plot import *


def mask_ops(image_np, resized_im, seg_map, pad_x):
    
    # Normalize the output
    labels = np.argmax(seg_map.squeeze(),-1)
    label = labels[:-pad_x]

    prediction_mask = (label.squeeze() == 7)

    # Let's apply some morphological operations to
    # create the contour for our sticker
    prediction_mask = np.invert(prediction_mask)
    
    cropped_object = image_np * np.dstack((prediction_mask,) * 3)
    
    square = skimage.morphology.square(5)

    temp = skimage.morphology.binary_erosion(prediction_mask, square)

    # Invert values
    negative_mask = (temp != True)
    
    eroding_countour = negative_mask * prediction_mask

    eroding_countour_img = np.dstack((eroding_countour, ) * 3)

    cropped_object[eroding_countour_img] = 248

    png_transparancy_mask = np.uint8(prediction_mask * 255)

    image_shape = cropped_object.shape

    png_array = np.zeros(shape=[image_shape[0], image_shape[1], 4], dtype=np.uint8)

    png_array[:, :, :3] = cropped_object

    png_array[:, :, 3] = png_transparancy_mask

    return png_array

def read_image(url):
    try:
        img = plt.imread(url)
    except IOError:
        print('Cannot retrieve image. Please check url: ' + url)
        return

    w, h, _ = img.shape

    ratio = 512. / np.max([w,h])
    image_np = cv2.resize(img, (int(ratio*h),int(ratio*w)))
    resized = image_np / 127.5 - 1.
    pad_x = int(512 - resized.shape[0])
    resized_im = np.pad(resized, ((0, pad_x),(0,0),(0,0)), mode='constant')
    return img, resized_im, pad_x


def main(url):
    deeplab_model = DeepModel()

    image, resized_im, _ = read_image(url)
    
    deeplab_model.mask_model.create_mask_tool.image = image
    model = deeplab_model.forward()
    seg_map = model.predict(np.expand_dims(resized_im, 0))
    input_mask = seg_map
    array = input_mask.squeeze()
    # Plot
    io.imshow(array)
    io.show()

if __name__ == '__main__':
    url = "./deeplab3/imgs/image1.jpg"
    main(url)
