from nets.mobilenetsv2 import SSD

import numpy as np
from keras.layers import Reshape, Lambda, Flatten, Activation, Conv2D, Conv2DTranspose, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
import keras.backend as K
import tensorflow as tf



def mask_gen():
    """SSD300 architecture.
    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.
    # References
        https://arxiv.org/abs/1512.02325
    """
    def crop_image(img, crop):
        return tf.image.crop_to_bounding_box(img,
                                crop[1],
                                crop[0],
                                crop[3] - crop[1],
                                crop[2] - crop[0])

    alpha=1.0
    img_size = (input_shape[1],input_shape[0])
    input_shape = (input_shape[1],input_shape[0],3)
    mobilenetv2_input_shape = (224,224,3)

    Input0 = Input(input_shape)

    ssd_mobilenetv2  = SSD(input_shape=mobilenetv2_input_shape,include_top=False, weights='imagenet')

    FeatureExtractor = Model(inputs=ssd_mobilenetv2.input, outputs=ssd_mobilenetv2.get_layer('predictions').output)

    x = FeatureExtractor(Input0) # Box loc, confidence, prior box

    # TODO OBTAIN THE MASK

    # img, mask =  MAGIC_BOX(x)



    cropping = Lambda(lambda x: K.map_fn(lambda y: crop_image(y[0], y[1]), elems=x, dtype=tf.float32),
                      output_shape=local_shape)
    
    l_img = cropping([g_img, in_pts])

    # TODO GET MASK
    
    
    
if __name__ == '__main__':
    from keras.utils import plot_model
    generator = model_generator()
    generator.summary()
    plot_model(generator, to_file='generator.png', show_shapes=True)
    discriminator = model_discriminator()
    discriminator.summary()
    plot_model(discriminator, to_file='discriminator.png', show_shapes=True)