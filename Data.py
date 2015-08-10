# -*- coding: utf-8 -*-
"""
Created on Mon May 18 02:28:21 2015

@author: Greg
"""
import numpy as np
import  h5py


class Data(object):
    def __init__(self, filename, num_images, batch_size, dim, start=0, seed=20150602):
        self.rng = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.dim = dim

        self.BUFF = 20
        with h5py.File(filename, 'r') as f:
            self.images = f['images'][start:start+num_images]
        self.num_images, imsize, imsize = self.images.shape
        self.imsize = imsize



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
        X = X/np.sqrt((X*X).sum(axis=1, keepdims=True))
        network.X.set_value(X.astype('float32'))
