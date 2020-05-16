#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras import backend as K
from keras.layers import Layer, Concatenate, Activation, LeakyReLU
import numpy as np
from utils import bilinear_sampler


# In[ ]:


class flowcomputation(Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        return 

    def call(self, inputs):
        
        if type(inputs) is not list or len(inputs) != 4:
            raise Exception('FlowComputation must be called on a list of four tensors [t, I0, I1, Optical_flow].'
                            'Instead got: ' + str(inputs))
        t =  inputs[0]
        I0 = inputs[1]
        I1 = inputs[2]
        Optical_flow = inputs[3]
        F01 = Optical_flow[:, :, :, :2]
        F10 = Optical_flow[:, :, :, 2:]
        
        Fhat_t0 = (t - 1.0) * t * F01 + (t * t) * F10
        Fhat_t1 = (1.0 - t) * (1.0 - t) * F01 - t * (1.0 - t) * F10
        
        g0 = g()([I0,Fhat_t0])
        g1 = g()([I1, Fhat_t1])
        
        return [Concatenate(axis = 3)([I0, I1, g0, g1, Fhat_t0, Fhat_t1]), F01, F10, Fhat_t0, Fhat_t1]
        
        
    def compute_output_shape(self, input_shape):
        output_channels = 3 * 4 + 2 * 2
        shape = list(input_shape[1])
        flow_shape = (shape[0], shape[1], shape[2], 2)
        
        return [(shape[0], shape[1], shape[2], output_channels),
                flow_shape, flow_shape, flow_shape, flow_shape]


# In[ ]:


class imagecomputation(Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        return 

    def call(self,inputs):
        
        if type(inputs) is not list or len(inputs) != 6:
            raise Exception('ImageComputation must be called on a list of six tensors [t, I0, I1, interpolation_output, Fhat_t0, Fhat_t1]'
                            'Instead got: ' + str(inputs))
        
        if K.int_shape(inputs[3])[3] != 5:
            raise Exception('ImageComputation must be with 5 lists in 4th dim'
                            'Instead got: ' + str(K.int_shape(inputs[3])[3]))
        t = inputs[0]
        I0 = inputs[1]
        I1 = inputs[2]
        Ft0 = inputs[3][:, :, :, :2] 
        Ft1 = inputs[3][:, :, :, 2:4]
        Vt0 = inputs[3][:, :, :, 4:5]
        
        Vt0 = Activation('sigmoid')(Vt0)
        Vt0 = K.tile(Vt0, (1, 1, 1, 3))
        Vt1 = 1.0 - Vt0

        Ft0 =  Ft0 + inputs[4]
        Ft1 =  Ft1 + inputs[5]
        
        g0 = g()([I0, Ft0])
        g1 = g()([I1, Ft1])
        
        epsilon = 1e-12  
        It = ((1.0 - t)* Vt0 * g0) + (t * Vt1 * g1)
        Z = ((1.0 - t) * Vt0 + t * Vt1) + epsilon
        It = It / Z
        
        return It
        
        
    def compute_output_shape(self, input_shapes):
        shape = list(input_shapes[1])
        return (tuple(shape))


# In[ ]:


class g(Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        return

    def call(self, inputs):
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception('ImageComputation must be called on a list of two tensors [Image, flow]'
                            'Instead got: ' + str(inputs))
            
        img = inputs[0]
        flow = inputs[1]
        return bilinear_sampler(img, flow)
    
    def compute_output_shape(self, input_shapes):
        shape = list(input_shapes[0])
        return (tuple(shape))

