#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import os
from random import randint, seed
import numpy as np


# In[17]:


class FrameGenerator():

    def __init__(self, height, width, filepath, rand_seed = None):
        
        self.height = height
        self.width = width
        self.filepath = filepath

        self.video_files = []
       
        filenames = [f for f in os.listdir(self.filepath)]
        self.video_files = [f for f in filenames if any(filetype in f.lower() for filetype in ['.mp4', '.flv', '.webm', '.mov', '.m4v', '.avi'])]
        print(">> Found {} video files in {}".format(len(self.video_files), self.filepath))  

        if rand_seed:
            seed(rand_seed)
    
    def extract_videoframes(self, seq_len, vid_obj):
        count = 0
        seq = []
        
        success , image = vid_obj.read()
        if not success or (image.shape[1] <= self.width) or (image.shape[0] <= self.height):
            return seq
        
        x = np.random.randint(0, image.shape[1] - self.width)
        y = np.random.randint(0, image.shape[0] - self.height)
         
        while count < (seq_len): 
            success, image = vid_obj.read()
            if not success:
                break
                
            image = image[y : (self.height + y), x : (self.width + x)]
            seq.append(image)
            count += 1
            
        return seq
    
    def sample(self, size, flip = False):
        sequence = []
        
        while(len(sequence) != size):
            file = os.path.join(self.filepath, np.random.choice(self.video_files, 1, replace = True)[0])
            vid_obj = cv2.VideoCapture(file)

            frame_count = vid_obj.get(cv2.CAP_PROP_FRAME_COUNT)
            start  = np.random.randint(0, np.absolute(frame_count - size))
            vid_obj.set(cv2.CAP_PROP_POS_FRAMES, start)

            sequence = self.extract_videoframes(size, vid_obj)
        
        if(flip):
            sequence = list(reversed(sequence))
            
        return sequence

