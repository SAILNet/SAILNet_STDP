# -*- coding: utf-8 -*-
"""
Created on Mon Mar 09 23:28:22 2015

@author: Greg
"""
import numpy as np
from Network import Network
import theano
import theano.tensor as T

class Activity():
    
    def get_acts(self,network):
        print 'cpu'
        
        B = network.X.dot(network.Q)
        T = np.tile(network.theta,(network.batch_size,1))
        Ys = np.zeros((network.batch_size,network.M))
        Y = np.zeros((network.batch_size,network.M))
        aas = np.zeros((network.batch_size,network.M))
        spike_train = np.zeros((network.batch_size,network.M,network.num_iterations))
        
        num_iterations = 50

        eta = .1
        
        for tt in xrange(num_iterations):
            Ys = (1.-eta)*Ys+eta*(B-aas.dot(network.W))
            aas = np.zeros((network.batch_size,network.M))
            #This resets the current activity of the time step to 0's        
            aas[Ys > T] = 1.
            #If the activity of a given neuron is above the threshold, set it to 1 a.k.a. fire.
            
            
            """        
            Second attempt at STDP, using more matricies     
            """
            spike_train[:,:,tt]=aas
            
            
            
            #Forces mean to be 0
            Y += aas
            #update total activity
            Ys[Ys > T] = 0.
            
        network.Y = Y
        network.spike_train = spike_train
        
class Activity_gpu():
    
    def __init__(self):
        X = T.matrix('X')
        Q = T.matrix('Q')
        theta = T.vector('theta')
        W = T.matrix('W')
        Y_o = T.matrix('Y')
        Ys_o = T.matrix('Ys')
        aas_o = T.matrix('aas')
        spike_train_o = T.tensor3('spike_train')

        Y = Y_o
        Ys = Ys_o
        aas = aas_o
        spike_train = spike_train_o

        B = X.dot(Q)
        Th = theta.dimshuffle('x', 0)

        num_iterations = 50
        eta = .1

        for tt in xrange(num_iterations):
            Ys = (1.-eta)*Ys+eta*(B-aas.dot(W))
            aas = 0.*aas
            #This resets the current activity of the time step to 0's        
            aas = T.switch(Ys > Th, 1., aas)
            #If the activity of a given neuron is above the threshold, set it to 1 a.k.a. fire.
            
            """        
            Second attempt at STDP, using more matricies     
            """
            spike_train = T.set_subtensor(spike_train[:,:,tt], aas)
            
            #Forces mean to be 0
            Y += aas
            #update total activity
            Ys = T.switch(Ys > Th, 0., Ys)

        self.f = theano.function([X, Q, theta, W, Y_o, Ys_o, aas_o, spike_train_o], [Y, spike_train])
        
    def get_acts(self,network):
        print 'gpu'
        
        Ys = np.zeros((network.batch_size,network.M)).astype('float32')
        Y = np.zeros((network.batch_size,network.M)).astype('float32')
        aas = np.zeros((network.batch_size,network.M)).astype('float32')
        spike_train = np.zeros((network.batch_size,network.M,network.num_iterations)).astype('float32')
        X = network.X.astype('float32')
        Q = network.Q.astype('float32')
        W = network.W.astype('float32')
        theta = network.theta.astype('float32')

        Y, spike_train = self.f(X, Q, theta, W, Y, Ys, aas, spike_train)
            
        network.Y = Y
        network.spike_train = spike_train
        
