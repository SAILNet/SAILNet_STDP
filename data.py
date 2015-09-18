# -*- coding: utf-8 -*-
"""
Created on Mon May 18 02:28:21 2015

@author: Greg
"""
import numpy as np
import  h5py


class Data(object):
    def __init__(self, filename, num_images, batch_size, dim, start=0, seed_or_rng=20150602):
        if isinstance(seed_or_rng,np.random.RandomState):
            self.rng = seed_or_rng
        else:            
            self.rng = np.random.RandomState(seed_or_rng)
        self.batch_size = batch_size
        self.dim = dim

        self.BUFF = 20
        with h5py.File(filename, 'r') as f:
            self.images = f['images'][start:start+num_images]
        self.num_images, imsize, imsize = self.images.shape
        self.imsize = imsize


class Static_Data(Data):
    def make_X(self, network):
        X = np.empty((self.batch_size, self.dim))
        sz = np.sqrt(self.dim).astype(np.int)
        for ii in xrange(self.batch_size):
            r = self.BUFF+int((self.imsize-sz-2.*self.BUFF)*self.rng.rand())
            c = self.BUFF+int((self.imsize-sz-2.*self.BUFF)*self.rng.rand())
            myimage = self.images[int(self.num_images*self.rng.rand()),r:r+sz,c:c+sz].ravel()
            #takes a chunck from a random image, size of 16X16 patch at a random location       
                
            X[ii] = myimage

        X = X-X.mean(axis=1, keepdims=True)
        #X = X/np.sqrt((X*X).sum(axis=1, keepdims=True))
        X = X/X.std(axis=1, keepdims=True)
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
            self.ims = self.rng.permutation(self.num_images)[:self.batch_size]
            self.locs = self.rng.randint(self.BUFF, self.imsize-self.BUFF,
                                         size=(self.batch_size, 2))
            # One of 9 directions
            self.dirs = self.rng.randint(-1, 2, size=(self.batch_size, 2))
        else:
            # Take step in direction
            self.locs += self.dirs

        for im, xy in zip(self.ims, self.locs):
            r, c = xy
            myimage = self.images[im, r:r+sz, c:c+sz].ravel()
                
            X[ii] = myimage

        X = X-X.mean(axis=1, keepdims=True)
        #X = X/np.sqrt((X*X).sum(axis=1, keepdims=True))
        X = X/X.std(axis=1, keepdims=True)
	network.X.set_value(X.astype('float32'))
        self.current_frame += 1
