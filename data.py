# -*- coding: utf-8 -*-
"""
Created on Mon May 18 02:28:21 2015

@author: Greg
"""
import numpy as np
import  h5py
import matplotlib.pyplot as plt

class Data(object):
    def __init__(self, filename, num_images, batch_size, dim, start=0, seed_or_rng=20150602,image_name = 'images'):
        if isinstance(seed_or_rng,np.random.RandomState):
            self.rng = seed_or_rng
        else:            
            self.rng = np.random.RandomState(seed_or_rng)
        self.batch_size = batch_size
        self.dim = dim

        self.BUFF = 20
        with h5py.File(filename, 'r') as f:
            self.images = f[image_name][start:start+num_images]
        self.num_images, imsize_x, imsize_y = self.images.shape
        self.imsize_x = imsize_x
        self.imsize_y = imsize_y

class Static_Data(Data):
    def make_X(self, network):
        X = np.empty((self.batch_size, self.dim))
        sz = np.sqrt(self.dim).astype(np.int)
        for ii in xrange(self.batch_size):
            r = self.BUFF+int((self.imsize_x-sz-2.*self.BUFF)*self.rng.rand())
            c = self.BUFF+int((self.imsize_y-sz-2.*self.BUFF)*self.rng.rand())
            myimage = self.images[int(self.num_images*self.rng.rand()), r:r+sz,c:c+sz].ravel()
            #takes a chunck from a random image, size of 16X16 patch at a random location       
                
            X[ii] = myimage

        X = X-X.mean(axis=1, keepdims=True)
        #X = X/np.sqrt((X*X).sum(axis=1, keepdims=True))
        X = X/X.std(axis=1, keepdims=True)
        assert not np.any(np.isnan(X))
        network.X.set_value(X.astype('float32'))

class Time_Data(Data):
    def __init__(self, filename, num_images, batch_size, dim, num_frames,
                 start=0, seed_or_rng=20150602):
        super(Time_Data,self).__init__(filename, num_images, batch_size,
                                       dim, start, seed_or_rng)
        self.num_frames = num_frames
        self.current_frame = 0
        self.BUFF += num_frames
        self.ims = None
        self.locs = None
        self.dirs = None

    def make_X(self, network):
        X = np.empty((self.batch_size, self.dim))
        sz = np.around(np.sqrt(self.dim)).astype(np.int)
        if self.ims is None or self.current_frame >= self.num_frames:
            network.initialize_time()
            self.current_frame = 0
            if self.ims is None:
                assert self.dirs is None
                assert self.locs is None
            # Choose random locations and directions
            if self.num_images >= self.batch_size:
                self.ims = self.rng.permutation(self.num_images)[:self.batch_size]
            else:
                self.ims = self.rng.randint(0, self.num_images, self.batch_size)
            self.locs_x = self.rng.randint(self.BUFF, self.imsize_x-self.BUFF-sz,size=(self.batch_size, 1))
            self.locs_y = self.rng.randint(self.BUFF, self.imsize_y-self.BUFF-sz,size=(self.batch_size, 1))
            self.locs = np.concatenate((self.locs_x,self.locs_y),axis=1)
            
            # One of 9 directions
            self.dirs = self.rng.randint(-1, 2, size=(self.batch_size, 2))
        else:
            # Take step in direction
            self.locs += self.dirs

        for ii, (im, xy) in enumerate(zip(self.ims, self.locs)):
            r, c = xy
            X[ii] = self.images[im, r:r+sz, c:c+sz].ravel()

        X = X-X.mean(axis=1, keepdims=True)
        #X = X/np.sqrt((X*X).sum(axis=1, keepdims=True))
        X = X/X.std(axis=1, keepdims=True)
        assert not np.any(np.isnan(X))
        if self.current_frame != 0:
            network.X_tm1.set_value(network.X.get_value())
        else:
            network.X_tm1.set_value(X.astype('float32'))
        network.X.set_value(X.astype('float32'))
        self.current_frame += 1
        
class Movie_Data(Data):
    def __init__(self, filename, num_images, batch_size, dim, num_frames,
                 start=0, seed_or_rng=20150602,image_name='m'):
        super(Movie_Data,self).__init__(filename, num_images, batch_size,
                                       dim, start, seed_or_rng,image_name)
        self.num_frames = num_frames
        self.current_frame = 0
        self.BUFF = 10
        self.ims = None
        self.locs = None
        self.dirs = None
        self.flag = 0

    def make_X(self, network):
        X = np.empty((self.batch_size, self.dim))
        sz = np.around(np.sqrt(self.dim)).astype(np.int)
        if self.ims is None or self.current_frame >= self.num_frames:
            network.initialize_time()
            self.current_frame = 0
            if self.ims is None:
                assert self.dirs is None
                assert self.locs is None
            # Choose random locations and directions
            if self.num_images >= self.batch_size:
                self.ims = self.rng.permutation(self.num_images-self.num_frames)[:self.batch_size]
            else:
                self.ims = self.rng.randint(0, self.num_images-20, self.batch_size)
            self.locs_x = self.rng.randint(self.BUFF, self.imsize_x-self.BUFF-sz,size=(self.batch_size, 1))
            self.locs_y = self.rng.randint(self.BUFF, self.imsize_y-self.BUFF-sz,size=(self.batch_size, 1))
            self.locs = np.concatenate((self.locs_x,self.locs_y),axis=1)
        else: #Move one step forward in time (1 image forward)
            self.ims += 1

        if self.flag == 0: #Save two images
            ex = self.images[30]
            ex2 = self.images[31]

            plt.imshow(ex,cmap='gray')
            plt.savefig('d30.png')

            plt.imshow(ex2,cmap='gray')
            plt.savefig('d31.png')


        for ii, (im, xy) in enumerate(zip(self.ims, self.locs)):
            r, c = xy
            X[ii] = self.images[im, r:r+sz, c:c+sz].ravel()

        X = X-X.mean(axis=1, keepdims=True)
        #X = X/np.sqrt((X*X).sum(axis=1, keepdims=True))
        X = X/X.std(axis=1, keepdims=True)
        
        Y = X[0].reshape((sz,sz))
        
        if self.flag == 0: #Saving movie sequence example
            plt.imshow(Y,cmap='gray')
            plt.savefig('duck'+str(self.current_frame)+'.png')
            if self.current_frame == 9:
                self.flag = 1

        assert not np.any(np.isnan(X))
        if self.current_frame != 0:
            network.X_tm1.set_value(network.X.get_value())
        else:
            network.X_tm1.set_value(X.astype('float32'))
        network.X.set_value(X.astype('float32'))
        self.current_frame += 1
