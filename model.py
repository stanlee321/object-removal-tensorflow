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
from keras.engine.topology import get_source_inputs

import numpy as np

# Tools
from tools.maskgen import CreateMaskTool

# Load deeplab model
from deeplab3.model import Deeplabv3
#from inpainting.model import model_generator
#from inpainting.model import model_discriminator


class MaskModel:
    def __init__(self, input_shape=(512, 512, 3), input_tensor=None, obj=7):
        self.obj = obj
        self.input_shape = input_shape
        self.create_mask_tool = CreateMaskTool(obj=obj)
    
    def call_py_function(self, x):
    
        img_np = x[:,:,:,0:3]
        mask   = x[:,:,:,3:]

        mask = tf.py_func(self.create_mask_tool.create_tf_mask, 
                    [img_np, mask],
                    [tf.float32],
                    stateful=False,
                    name='mask_opt')
        return mask

    def forward(self, input_tensor=None):
        if input_tensor is None:
            img_input = Input(shape=self.input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=self.input_shape)
            else:
                img_input = input_tensor
        # x = MaskLayer(input_shape=input_shape, output_dim=input_shape)(img_input)
        x = Lambda(self.call_py_function, output_shape=K.get_variable_shape(img_input))(img_input)
        
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
        
        model = Model(inputs, x)

        return model


class DeepModel:
    def __init__(self, input_tensor=None, input_shape=(512, 512, 3), obj=7):
        self.input_tensor = input_tensor
        self.input_shape = input_shape
        self.mask_model = MaskModel(input_shape=self.input_shape, obj=obj)
        self.deeplab_model = Deeplabv3(weights='pascal_voc', 
                                        input_tensor=None,
                                        input_shape=self.input_shape, 
                                        classes=21, 
                                        backbone='mobilenetv2', 
                                        OS=16, alpha=1.)

    def forward(self, input_tensor=None):
        # Set Tensor Input Placeholder
        if input_tensor is None:
            img_input = Input(shape=self.input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=self.input_shape)
            else:
                img_input = input_tensor

        # Concatenate DeepLab Mask Outout + Input
        x = keras.layers.concatenate([img_input, self.deeplab_model(img_input)])

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create deeplab + masked_output
        mask_model = self.mask_model.forward()

        model = Model(inputs, mask_model(x))
        
        return model
    
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