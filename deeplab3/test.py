from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import skimage.morphology

from model import Deeplabv3
import skimage.morphology
import skimage.io as io
from skimage.color import rgba2rgb



def mask_ops(image_np, resized_im, seg_map, pad_x):
    
    # Normalize the output
    labels = np.argmax(seg_map.squeeze(),-1)
    label = labels[:-pad_x]
    print('paddd', pad_x)
    print('LABEL...', label.shape)
    prediction_mask = (label.squeeze() == 7)
    print('Preddd...', prediction_mask.shape)

    prediction_mask = np.invert(prediction_mask)

    # Let's apply some morphological operations to
    # create the contour for our sticker
    
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
    image_np = cv2.resize(img,(int(ratio*h),int(ratio*w)))
    resized = image_np / 127.5 - 1.
    pad_x = int(512 - resized.shape[0])
    resized_im = np.pad(resized,((0,pad_x),(0,0),(0,0)), mode='constant')

    return image_np, resized_im, pad_x




def main(url):
    # Plot the image with holes

    deeplab_model = Deeplabv3()

    image_np, resized_im, pad_x = read_image(url)
    
    seg_map = deeplab_model.predict(np.expand_dims(resized_im, 0))

    input_mask = mask_ops(image_np, resized_im, seg_map, pad_x)

    print(input_mask.shape)
    input_mask = rgba2rgb(input_mask)

    c = tf.convert_to_tensor(input_mask)
    print(c.get_shape().as_list())

    with tf.Session()  as sess:
        array = c.eval(session=sess)

    # Plot
    io.imshow(array)
    io.show()

    ##################################
    # Differente plot options
    ##################################
    # from plot import *

    # Plot the 3 column image
    # run_visualization(url, deeplab_model)

    # Plot simple image with mask
    # simple_plot(deeplab_model, url)

    # cv2.imwrite('image_labels.jpg', labels[:-pad_x])

if __name__ == '__main__':
    url = "imgs/image1.jpg"
    main(url)
