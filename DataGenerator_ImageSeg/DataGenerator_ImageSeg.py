import os
import numpy as np
import random
from numpy.random import shuffle
import gc

class DataGenerator_ImageSeg():
    'Generates augmented batch of data'
    def __init__(self, path_data, path_labels, batch_size=32, patch_size = 128, n_of_band = 4,
                 n_of_classes = 3, augment=True, shuffle = True):
        'Initialization'
        self.path_data = path_data
        self.path_labels = path_labels
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.n_of_band = n_of_band
        self.n_of_classes = n_of_classes
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()
        self.available_batches = self.get_available_batches()

        gc.collect

    def on_epoch_end(self):
        'Creates a list of all available indexes'
        temp_list = []
        self.list_of_file_of_data = [] #list of all available npy file
        self.list_of_file_of_labels = [] #list of all available npy file
        self.list_of_patch = [] #list of number of patch in each npy file

        os.chdir(self.path_labels)
        temp_list = os.listdir()
        for l in temp_list:
            if l.endswith('npy'):
                temp = np.load(l).shape[0]
                if temp >= self.batch_size:
                    self.list_of_file_of_labels.append(l)

        os.chdir(self.path_data)
        temp_list = os.listdir()
        for l in temp_list:
            if l.endswith('npy'):
                temp = np.load(l).shape[0]
                if temp >= self.batch_size:
                    self.list_of_file_of_data.append(l)                

        for i in range(0,len(self.list_of_file_of_data)):
            temp = np.load(self.list_of_file_of_data[i]).shape[0]
            self.list_of_patch.append(list(range(0,temp)))
            if self.shuffle == True:
                shuffle(self.list_of_patch[i])
        
        gc.collect
 
    
    def get_status(self):
        'print status of dataset'
        empty = True
        for i in range(0,len(self.list_of_file_of_data)):
            print('File: - {} - with {} remaining patches\nExtended status is:\n {}'
            .format(self.list_of_file_of_data[i],len(self.list_of_patch[i]),self.list_of_patch[i]))
            empty = False
        if empty:
            print('The whole dataset has been explored. (EMPTY)')
        
        gc.collect

    def patch_extractor(self):
        'return a cople of list of data and tare with size (batch_size, dimx, dimy, band)'
        os.chdir(self.path_labels)
        temp_load_label = np.load(self.list_of_file_of_labels[0])
        os.chdir(self.path_data)
        temp_load_data = np.load(self.list_of_file_of_data[0])

        X = np.empty((self.batch_size, self.patch_size ,self.patch_size , self.n_of_band), dtype = np.float32)
        y = np.empty((self.batch_size, self.patch_size ,self.patch_size , self.n_of_classes), dtype = np.float32)
        

        X = temp_load_data[self.list_of_patch[0][0:self.batch_size],:,:,:]
        
        y = temp_load_label[self.list_of_patch[0][0:self.batch_size],:,:,:]

        if self.augment == True:
            for i in range(0,self.batch_size):
                X[i,:,:,:], y[i,:,:,:] = self.__data_augmentation(X[i,:,:,:], y[i,:,:,:])

        del self.list_of_patch[0][0:self.batch_size]
        
        if (len(self.list_of_patch[0])) < self.batch_size:
            del self.list_of_file_of_data[0]
            del self.list_of_file_of_labels[0]
            del self.list_of_patch[0]

        gc.collect

        return X, y

            
    def __data_augmentation(self, X, y):
        'Data augmentation based on flips and rotations'
        fliplr = bool(random.getrandbits(1))
        flipud = bool(random.getrandbits(1))
        rot90 = random.randint(0,3)

        if fliplr:
            X = np.fliplr(X)
            y = np.fliplr(y)            
        if flipud:
            X = np.flipud(X)
            y = np.flipud(y)

        X = np.rot90(X,k=rot90)
        y = np.rot90(y,k=rot90)  
        
        gc.collect()

        return X, y

    def get_available_batches(self):
        self.available_batches = 0
        for i in range(0,len(self.list_of_file_of_data)):
            module = len(self.list_of_patch[i]) % self.batch_size
            self.available_batches = int(self.available_batches + (len(self.list_of_patch[i]) - module) /self.batch_size)
        
        return self.available_batches