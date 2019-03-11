import numpy as np
import tensorflow as tf
import cv2
import skimage.morphology
import skimage.io as io
from skimage.color import rgba2rgb


class CreateMaskTool:
    def __init__(self):
        self.image = None
        # Setter/Getter for image
    @property
    def image(self):
        return self.__image
    @image.setter
    def image(self, new_image):
        self.__image = new_image

    def _mask_ops(self, resized_im, seg_map, pad_x):
        # Normalize the output
        labels = np.argmax(seg_map,-1)
        label = labels[:-pad_x]
        prediction_mask = (label.squeeze() == 7)

        # Let's apply some morphological operations to
        # create the contour for our sticker
        prediction_mask = np.invert(prediction_mask)
        
        cropped_object = resized_im * np.dstack((prediction_mask,) * 3)
        
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


    def _get_pad(self):
        w, h, _ = self.image.shape

        ratio = 512. / np.max([w,h])
        image_np = cv2.resize(self.image, (int(ratio*h),int(ratio*w)))
        resized = image_np / 127.5 - 1.
        pad_x = int(512 - resized.shape[0])
        resized_im = np.pad(resized, ((0, pad_x),(0,0),(0,0)), mode='constant')

        return image_np, resized_im, pad_x

    def create_tf_mask(self, input_image_np, tensor_mask_pred):
        """

        input_image_np: WxHx3 channel image
        tensor_mask_pred: Output  Tensor from the last layer 

        """

        print('INPUT TENSOR_IMAGENP', input_image_np.shape)
        print('INPUT MASK TENSOR', tensor_mask_pred.shape)

        input_image_np = input_image_np.squeeze()
        tensor_mask_pred = tensor_mask_pred.squeeze()
        print('....')
        print('INPUT TENSOR_IMAGENP', input_image_np.shape)
        print('INPUT MASK TENSOR', tensor_mask_pred.shape)

        image_np, resized_im, pad_x = self._get_pad()

        # Create mask image with white holes
        input_mask = self._mask_ops(image_np, tensor_mask_pred, pad_x)
        input_mask = input_mask.astype(np.float32)
        # Add batch channel and convert to tensor
        batch_mask = np.expand_dims(input_mask,0)
        #tf_mask = tf.convert_to_tensor(batch_mask)
    
        #print('OUT TESNOR')
        #print(tf_mask.get_shape())
        return batch_mask