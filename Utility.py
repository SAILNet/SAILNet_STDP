# -*- coding: utf-8 -*-
"""
Created on Mon May 18 02:28:21 2015

@author: Greg
"""
import numpy as np

rng = np.random.RandomState(0)

BUFF = 20

def make_X(network, images):
    num_images, imsize, imsize = images.shape
    X = np.zeros(network.X.get_value().shape)
    sz = np.sqrt(network.N).astype(np.int)
    for ii in xrange(network.batch_size):
        r = BUFF+int((imsize-sz-2.*BUFF)*rng.rand())
        c = BUFF+int((imsize-sz-2.*BUFF)*rng.rand())
        myimage = images[int(num_images*rng.rand()),r:r+sz,c:c+sz].ravel()
        #takes a chunck from a random image, size of 16X16 patch at a random location       
            
            
        X[ii] = myimage
        
    return X