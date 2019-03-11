import numpy as np
import tensorflow as tf
import cv2
import skimage.morphology
import skimage.io as io
from skimage.color import rgba2rgb

def _mask_ops(image_np, resized_im, seg_map, pad_x):
    
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

    input_mask = rgba2rgb(png_array)

    return input_mask


def _norm_input_image(image_np):
    #raise ValueError('label value too large.')
        
    w, h, _ = image_np.shape
    ratio = 512. / np.max([w,h])
    resized_im = cv2.resize(image_np,(int(ratio*h),int(ratio*w)))
    resized = resized_im / 127.5 - 1.
    pad_x = int(512 - resized.shape[0])
    norm_img = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')

    return resized_im, norm_img, pad_x



def create_tf_mask(input_image_np, tensor_mask_pred):
    """

    input_image_np: WxHx3 channel image
    tensor_mask_pred: Output  Tensor from the last layer 

    """

    resized_im, norm_image, pad_x = _norm_input_image(input_image_np)

    # Create mask image with white holes
    input_mask = _mask_ops(resized_im, norm_image, tensor_mask_pred, pad_x)

    # Add batch channel and convert to tensor
    batch_mask = np.expand_dims(input_mask,0)
    tf_mask = tf.convert_to_tensor(batch_mask)

    return tf_mask