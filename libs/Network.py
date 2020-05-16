#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import sys
# sys.path.insert(1,r'C:\Users\sb computer\Jupyter Notebook\NVIDIA-SuperSlowMo\utils')
# sys.path.insert(1,r'C:\Users\sb computer\Jupyter Notebook\NVIDIA-SuperSlowMo\libs')


# In[27]:


import numpy as np
from Layers import flowcomputation, imagecomputation, g
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D,LeakyReLU, AveragePooling2D, Activation
from keras.layers.merge import Concatenate
from keras.applications import VGG16
from keras import backend as K
from DataGenerator import FrameGenerator
import os

# In[28]:


class Network(object):
    def __init__(self, shape = (512, 512, 3), vgg_model = None, lr = 0.0001):
        self.shape = shape
        
        self.vgg = self.load_vgg(vgg_model)
        self.model, outputs =  self.SlowMo_network()        
        self.current_epoch = 0
        self.compile_network(self.model, outputs, lr = lr)
    
    def load_vgg(self, vgg_weights):
            
        img = Input(self.shape)
#         mean = [0.485, 0.456, 0.406]
#         sd = [0.229, 0.224, 0.225]
        
#         processed = Lambda(lambda x: (x-mean) / sd)(img)
        vgg_layers = [13]

        if vgg_weights:
            vgg = VGG16(weights = None, include_top = False)
            vgg.load_weights(vgg_weights, by_name = True)
        
        else:
            vgg = VGG16()
        
        vgg.outputs = [vgg.layers[i].output for i in vgg_layers]
        out = vgg(img)
        
        model = Model(inputs = img, outputs = out)
        model.trainable = False
        model.compile(loss='mse', optimizer='adam')
        
        return model
    
    
    def UNet(self, I, output_channels, kernel_size = [7,5,3,3,3,3], decoder_extra_input = None, alpha = 0.1, 
             return_activations = False, apply_activation = False):
        
        def encode_layer(I, filters, kernel_size, downsampling = True):
          
            if downsampling:
                features = AveragePooling2D((2,2), strides = 2)(I)
            else:
                features = I

            conv1  = Conv2D(filters, kernel_size, strides = 1, padding = 'same')(features)
            act1 = LeakyReLU(alpha)(conv1)

            conv2  = Conv2D(filters, kernel_size, strides = 1, padding = 'same')(act1)
            out = LeakyReLU(alpha)(conv2)

            return out

        def decode_layer(I, concat_I, filters, kernel_size):
            
            upsampled_I = UpSampling2D(size = (2,2))(I)
            conv1 = Conv2D(filters, kernel_size, strides = 1, padding = 'same')(upsampled_I)
            act1 = LeakyReLU(alpha)(conv1)
            
            concat_I = Concatenate(axis = 3)([act1, concat_I])
            conv2  = Conv2D(filters, kernel_size, strides = 1, padding = 'same')(concat_I)
            out = LeakyReLU(alpha)(conv2)
            
            return out
            
        encodings = []
        encodings.append(I)
        
        encodings.append(encode_layer(encodings[0], 32,  kernel_size[0], False))
        encodings.append(encode_layer(encodings[1], 64,  kernel_size[1]))
        encodings.append(encode_layer(encodings[2], 128, kernel_size[2]))
        encodings.append(encode_layer(encodings[3], 256, kernel_size[3]))
        encodings.append(encode_layer(encodings[4], 512, kernel_size[4]))
        encodings.append(encode_layer(encodings[5], 512, kernel_size[5]))
        
        decodings = encodings[6]
        
        if decoder_extra_input is not None:
            decodings = Concatenate(axis = 3)([decodings,decoder_extra_input])
        
        decodings = decode_layer(decodings, encodings[5], 512, kernel_size[4])
        decodings = decode_layer(decodings, encodings[4], 256, kernel_size[3])
        decodings = decode_layer(decodings, encodings[3], 128, kernel_size[2])
        decodings = decode_layer(decodings, encodings[2], 64,  kernel_size[1])
        decodings = decode_layer(decodings, encodings[1], 32,  kernel_size[0])
        
        out = Conv2D(output_channels, 1)(decodings)
        
        if apply_activation:
            out = LeakyReLU(alpha)(out)
             
        if return_activations:
            return [encodings[6],out]
        else:
            return out
        
    def SlowMo_network(self, t = 0.5):
        
        I0 = Input(self.shape)
        I1 = Input(self.shape)
        t = Input((1,1,1))

        flow_computation_input = Concatenate(axis = 3)([I0,I1])
        
        encoding, Optical_flow = self.UNet(I = flow_computation_input, output_channels = 4, return_activations = True,
                                           apply_activation = True)
        
                
        flow_interpolation_input, F01, F10, Fhat_t0, Fhat_t1 =  flowcomputation()([t, I0, I1, Optical_flow])
        
        flow_interpolation_output = self.UNet(I = flow_interpolation_input, kernel_size = [3,3,3,3,3,3],
                                              decoder_extra_input = encoding, output_channels = 5, apply_activation = True)
        
        It = imagecomputation()([t, I0, I1, flow_interpolation_output, Fhat_t0, Fhat_t1])
                
        model = Model(inputs = [t, I0, I1], output = It)
                
        return [model, (I0, I1, F01, F10, Fhat_t0, Fhat_t1)]
    
    
    def compile_network(self, model, supp = None, lr = 0.0001):
        self.model.compile(optimizer = Adam(lr = lr), loss = self.total_loss(supp), metrics=[self.PSNR])
        
    def total_loss(self, outputs):
        I0, I1, F01, F10, Fhat_t0, Fhat_t1 = outputs
    
        def l1_loss(y_pred, y_true):
            return K.mean(K.abs(y_pred - y_true), axis=[1, 2, 3])

        def l2_loss(y_pred, y_true):
            return K.mean(K.square(y_pred - y_true), axis = [1, 2, 3])
        

        def reconstruction_loss(y_pred, y_true):
            return l1_loss(y_pred, y_true)


        def perceptual_loss(y_pred, y_true):
            y_pred = self.vgg(y_pred)
            y_true = self.vgg(y_true)
#             loss = 0
#             for pred, true in zip(y_pred, y_true):
#                 loss += l2_loss(pred, true)
            return l2_loss(y_pred, y_true)


        def wrapping_loss(frame0, frame1, frameT, F01, F10, Fdasht0, Fdasht1):
            return l1_loss(frame0, g()([frame1, F01])) +                    l1_loss(frame1, g()([frame0, F10])) +                    l1_loss(frameT, g()([frame0, Fdasht0])) +                    l1_loss(frameT, g()([frame1, Fdasht1]))


        def smoothness_loss(F01, F10):
            deltaF01 = K.mean(K.abs(F01[:, 1:, :, :] - F01[:, :-1, :, :])) + K.mean(
                K.abs(F01[:, :, 1:, :] - F01[:, :, :-1, :]))
            deltaF10 = K.mean(K.abs(F10[:, 1:, :, :] - F10[:, :-1, :, :])) + K.mean(
                K.abs(F10[:, :, 1:, :] - F10[:, :, :-1, :]))
            return (deltaF01 + deltaF10)
    
        def loss(y_true, y_pred):
            lr = reconstruction_loss(y_pred, y_true)
            lw = wrapping_loss(I0, I1, y_pred, F01, F10, Fhat_t0, Fhat_t1)
            lp = perceptual_loss(y_pred, y_true)
            ls = smoothness_loss(F01, F10)   
            return (255.0 * 0.8) * lr + 0.005 * lp + (255 * 0.4) * lw + 1 * ls

        return loss


    def PSNR(self, y_true, y_pred):
        mse = K.mean(K.square(y_true - y_pred))
        epsilon = 1e-10
        return  10.0 * K.log(1.0 / (mse + epsilon)) / K.log(10.0)    
       
    def summary(self):
        print(self.model.summary())
          
    def predict(self, sample, **kwargs):
        return self.model.predict(sample, **kwargs)

        
    def load(self, filepath, lr = 0.0001):

        self.model, supplementary = self.SlowMo_network()
        self.compile_network(self.model, supplementary, lr = lr)
        epoch = int(os.path.basename(filepath).split('.')[1].split('-')[0])
        assert epoch > 0, "Could not parse weight file. Should include the epoch"
        self.current_epoch = epoch
        self.model.load_weights(filepath)
        
    def fit_generator(self, generator, *args, **kwargs):
        self.model.fit_generator(generator,*args, **kwargs)


# In[29]:


#X = Network(vgg_model = 'vgg16.h5')

