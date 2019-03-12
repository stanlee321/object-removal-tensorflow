from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import skimage.morphology

from model import DeepModel
import skimage.morphology
import skimage.io as io
from skimage.color import rgba2rgb

"""
{
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'potted-plant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tv/monitor',
    255: 'ambigious'
}
"""

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
    deeplab_model = DeepModel(obj=7)

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
