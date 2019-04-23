"""
DATA GENERATOR FOR KERAS
"""
import numpy as np
import keras
from keras.utils import to_categorical
import random
import cv2
import os
import skvideo.io
import skvideo
skvideo.setFFmpegPath('C:\\ffmpeg-4.0.2-win64-static\\bin') #Configure this variable in your Window's 
                                                            #Path variable or read the videos frame per frame using OpenCV

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size=64, frames=30, dim=224, n_channels=3, path='', classes=60,
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.frames = frames
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.path = path
        self.classes = classes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def randomcrop(self, img, height, width):
        assert img.shape[1] >= height
        assert img.shape[2] >= width
        x = random.randint(0, img.shape[2] - width)
        y = random.randint(0, img.shape[1] - height)
        img = img[:,y:y + height, x:x + width,:]
        return img
        
    def padborder(self,img,dim):

        [C,M,N,P] = img.shape
        img2 = np.zeros((C, dim, dim, P), dtype=np.uint8)
        if M>N:
            I2=np.zeros((C,M,M,P),dtype=np.uint8)
            for i in range(0,C):
                I2[i,:,int(np.floor((M-N)/2)):int(np.floor((M-N)/2)+N),:]=img[i,:,:,:]
                img2[i,:,:,:] = cv2.resize(I2[i,:,:,:],(dim,dim),interpolation=cv2.INTER_AREA)
        elif N>M:
            I2 = np.zeros((C, N, N, P), dtype=np.uint8)
            for i in range(0, C):
                I2[i, int(np.floor((N - M)/2)):int(np.floor((N - M)/2)+M), :, :] = img[i, :, :, :]
                img2[i,:,:,:] = cv2.resize(I2[i,:,:,:], (dim, dim), interpolation=cv2.INTER_AREA)
        else:
            I2 = img
            for i in range(0, C):
                img2[i,:,:,:] = cv2.resize(I2[i,:,:,:], (dim, dim), interpolation=cv2.INTER_AREA)

        return img2

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, dim, dim, n_channels)
        
        # Initialization
        X = np.empty((self.batch_size, self.frames, self.dim, self.dim,self.n_channels),dtype=np.uint8)
        yc = np.empty(self.batch_size)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            addr = self.path + ID[len(self.path):]
            
            # Read video using skvideo
            img = skvideo.io.vread(addr)
            
            # Cut a central of the frame and resize it to the desired dimensions 
            for e in range(0, self.frames):
                X[i,e] = cv2.resize(img[e, 90:928,388:1425, :],(self.dim,self.dim)).astype(np.uint8)
                
            # NOTE: If you know the coordinates of the action tube (ROI) a priori, you should cut the frames like this:
            # Load ROI
            # addrROI =  self.path + '//ROI' + ID[len(self.path):-4] + '_roi.npy'
            # at = np.load(addrROI) # at = (xmin,ymin,xmin,ymax)
            # Cuts, adds padding (adds black borders to maintain the aspect ratio) and resample action tube
            # img2 = self.padborder(img[:,at[0]:at[2],at[1]:at[3],:], self.dim)
            
            # Gets the index of the action (from 1 to 60)
            yc[i] = int(addr[len(self.path)+18:len(self.path)+21])-1
        
        # Converts the class vector (integers) to binary class matrix.
        y = to_categorical(yc,self.classes)

        return X, y
