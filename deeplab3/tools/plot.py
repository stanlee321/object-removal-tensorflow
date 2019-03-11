import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from io import BytesIO
from PIL import Image
import cv2 # used for resize. if you dont have it, use anything else
import skimage.io as io
import skimage.morphology


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    print('shape colormap', colormap.shape)
    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
    print('label shape before::', label.shape)
    value = colormap[label]
    print('value shape::', value.shape)
    return value


def vis_segmentation(image, label):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(label).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.2)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(label)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()

def run_model(model, resized2):
    seg_map = model.predict(np.expand_dims(resized2,0))
    return seg_map

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
    resized_im = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')

    return image_np, resized_im, pad_x

def simple_plot(model, url):

    # Read Inputs
    image_np, resized_im, pad_x = read_image(url)

    # Predict mask map
    seg_map = run_model(model, resized_im)

    # Normalize the output
    labels = np.argmax(seg_map.squeeze(),-1)
    label = labels[:-pad_x]

    # Plot
    io.imshow(label.squeeze())
    io.show()


def run_visualization(url, MODEL):
    """Inferences DeepLab model and visualizes result."""
    try:
        img = plt.imread(url)
    except IOError:
        print('Cannot retrieve image. Please check url: ' + url)
        return

    w, h, _ = img.shape
    ratio = 512. / np.max([w,h])
    resized_l = cv2.resize(img,(int(ratio*h),int(ratio*w)))
    resized = resized_l / 127.5 - 1.
    pad_x = int(512 - resized.shape[0])
    resized_im = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')

    seg_map = run_model(MODEL, resized_im)
    print(seg_map.shape)
    labels = np.argmax(seg_map.squeeze(),-1)
    label = labels[:-pad_x]
    print(label.shape)
    vis_segmentation(resized_l, label)

def pascal_segmentation_lut():
    return {
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
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
