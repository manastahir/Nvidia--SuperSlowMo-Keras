#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import sys
# sys.path.insert(0,r'C:\Users\sb computer\Jupyter Notebook\NVIDIA-SuperSlowMo\libs')
# sys.path.insert(0,r'C:\Users\sb computer\Jupyter Notebook\NVIDIA-SuperSlowMo\utils')


# In[2]:


import numpy as np
from Layers import flowcomputation, imagecomputation, g
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D,LeakyReLU, AveragePooling2D, Activation
from keras.layers.merge import Concatenate
from keras.applications import VGG16
from keras import backend as K
import time

# In[3]:


class Network(object):
    def __init__(self, shape = (512, 512, 3), vgg_model = None, lr = 0.0001):
        self.shape = shape
        
        self.vgg = self.load_vgg(vgg_model)
        self.flow_model, self.interpolation_model, self.model, outputs =  self.SlowMo_network()        
        
        self.compile_network(self.flow_model)
        self.compile_network(self.interpolation_model)
        self.compile_network(self.model, outputs, lr = lr)
    
    def load_vgg(self, vgg_weights):
            
        img = Input(self.shape)
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
        
    def SlowMo_network(self):
        
        I0 = Input(self.shape)
        I1 = Input(self.shape)
        
        
        flow_computation_input = Concatenate(axis = 3)([I0,I1])
        
        enc, flow = self.UNet(I = flow_computation_input, output_channels = 4, return_activations = True,
                              apply_activation = True)
        
        flow_model = Model(inputs = [I0, I1], outputs = [flow, enc])
        
        
        flow_shape = K.int_shape(flow)
        enc_shape = K.int_shape(enc)
        
        Optical_flow = Input(flow_shape[1:])
        encoding = Input(enc_shape[1:])
        t = Input((1,1,1))
                
        flow_interpolation_input, F01, F10, Fhat_t0, Fhat_t1 =  flowcomputation()([t, I0, I1, flow])
        
        flow_interpolation_output = self.UNet(I = flow_interpolation_input, kernel_size = [3,3,3,3,3,3],
                                              decoder_extra_input = enc, output_channels = 5, apply_activation = True)
        
        It = imagecomputation()([t, I0, I1, flow_interpolation_output, Fhat_t0, Fhat_t1])
        
        interpolation_model = Model(inputs = [I0, I1, t, Optical_flow, encoding], outputs = It)
        
        
#         Optical_flow, encoding = flow_model([I0, I1])
#         It = interpolation_model([I0, I1, t, Optical_flow, encoding])
        model = Model(inputs = [t, I0, I1], output = It)
                
        return [flow_model, interpolation_model, model, (I0, I1, F01, F10, Fhat_t0, Fhat_t1)]
    
    
    def compile_network(self, model, supp = None, lr = 0.0001):
        if supp is not None:
            model.compile(optimizer = Adam(lr = lr), loss = self.total_loss(supp), metrics=[self.PSNR])
        else:
            model.compile(optimizer = 'adam', loss = 'mse')
        
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
        return  10.0 * K.log(1.0 / mse) / K.log(10.0)    
       
    def summary(self, model = 0):
        if model == 0:
            print(self.model.summary())
        elif model == 1: 
            print(self.flow_model.summary())
        elif model == 2: 
            print(self.interpolation_model.summary())    
            
    def predict(self, in_frames, fps_factor, **kwargs):
        I0, I1 = in_frames
        batch = len(I0)
        
        Optical_flow, encoding = self.flow_model.predict(in_frames, **kwargs)
        
        t = [(1.0 / fps_factor) * i for i in range(1, fps_factor, 1)]
        T = [np.full((batch,1,1,1), ti) for ti in t]
        frames = []
        
        for i in range(len(T)):
            f = self.interpolation_model.predict([I0,I1, T[i], Optical_flow, encoding])
            frames.append(f)
        
        return frames

        
    def load(self, filepath, lr = 0.0001):
        self.flow_model,self.interpolation_model,self.model, supplementary = self.SlowMo_network()
        self.compile_network(self.flow_model)
        self.compile_network(self.interpolation_model)
        self.compile_network(self.model, supplementary, lr = lr)
        self.model.load_weights(filepath)
        
    def fit_generator(self, generator, *args, **kwargs):
        self.model.fit_generator(generator,*args, **kwargs)


# In[4]:


# X = Network(vgg_model = 'vgg16.h5')


# In[5]:


# from DataGenerator import FrameGenerator


# In[6]:


# class DataFrameGenerator():
#     def __init__(self, directory, height, width, batch_size, rescale = 1.0, seed = None):
#         self.batch_size = batch_size
#         self.frame_generator = FrameGenerator(height, width, directory, seed)
#         self.rescale = rescale
#         self.fps = [2, 4, 8, 8, 16, 16, 32, 32]
        
#     def flow_from_directory(self, fps = None):
      
#       while True:
#         if fps is None:
#             fps = np.random.choice(self.fps, 1, replace = True)[0]

#         frame = np.random.randint(1, fps)
#         t = (1.0 / fps) * frame

#         frames = self.frame_generator.sample(self.batch_size + fps)

#         I0 = []
#         I1 = []
#         It = []

#         frames = [f * self.rescale for f in frames]
#         mean = [0.397, 0.431, 0.429]
#         sd  = [1, 1, 1]
#         frames = [(f-mean)/sd for f in frames]

#         for i in range(self.batch_size):
#             I0.append(frames[i])
#             I1.append(frames[i + fps])
#             It.append(frames[i + frame])

#         I0 = np.asarray(I0)
#         I1 = np.asarray(I1)
#         It = np.asarray(It)
#         t = np.full((self.batch_size,1,1,1), t)
#         yield [t, I0, I1], It


# In[7]:


# train_datagen = DataFrameGenerator('E:\Compressed\DeepVideoDeblurring_Dataset_Original_High_FPS_Videos\original_high_fps_videos',
#                                    512, 512, 6, rescale = 1.0/255)
# train_generator = train_datagen.flow_from_directory()


# In[8]:


# X.fit_generator(train_generator, steps_per_epoch = 10, epochs = 10)


# In[9]:


# sample, _ = next(train_generator)
# t, I0, I1 = sample


# In[ ]:


# frames = X.predict([I0, I1], 4)


# In[ ]:


# len(frames[0])


# In[ ]:


# frames[0][0].shape

