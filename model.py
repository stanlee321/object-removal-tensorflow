from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import Layer
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dense
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Sequential
from keras.models import Model
from keras.utils import plot_model
import keras
import keras.backend as K
import tensorflow as tf

import numpy as np

# Tools
from tools.maskgen import create_tf_mask

# Load deeplab model
from deeplab3.model import Deeplabv3
from inpainting.model import model_generator
from inpainting.model import model_discriminator


def call_py_function(x):

    print(x.get_shape())

    mask = x[:,:,:,0:3]
    img_np = x[:,:,:,3:]

    print('MASK', mask.get_shape())
    print('img-np', img_np.get_shape())

    mask = tf.py_func(create_tf_mask, 
                [img_np, mask],
                ["float32"],
                stateful=False,
                name='mask_opt')
    return mask

def MaskModel(input_shape=(512, 512, 3), input_tensor=None):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    inputs = img_input

    # x = MaskLayer(input_shape=input_shape, output_dim=input_shape)(img_input)
    x = Lambda(call_py_function, output_shape=input_shape)(inputs)

    model = Model(inputs, x)

    return model


def DeepModel(input_shape=(512, 512, 3)):

    # Set Tensor Input Placeholder
    inputs = Input(shape=input_shape, name='deeplab_input')

    # Instantiate DeepLabv3 model
    deeplab_model = Deeplabv3(weights='pascal_voc', 
                        input_tensor=None,
                        input_shape=input_shape, 
                        classes=3, 
                        backbone='mobilenetv2', 
                        OS=16, alpha=1.)

    # Instantiate MaskModel Model
    mask_model = MaskModel(input_shape=input_shape)

    # Concatenate DeepLab Mask Outout + Input
    x = keras.layers.concatenate([inputs, deeplab_model(inputs)])

    # Create deeplab + masked_output
    deeplab_seg_model = Model(inputs, mask_model(x))
    
    return deeplab_seg_model
    
    """
    # Instantiate Generator model
    generator = model_generator(input_shape)

    # Create Generator model
    generator_seg_model = Model(inputs, generator(deeplab_seg_model(inputs)))

    # Instatiate Discrimitator model
    
    plot_model(generator_seg_model, to_file='generator_deeplab.png', show_shapes=True)
    """

    #autoencoder = Model(inputs, decoder(deeplab(inputs)), name='autoencoder')    
if __name__ == '__main__':
    DeepModel()