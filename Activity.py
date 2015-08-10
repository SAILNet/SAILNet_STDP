# -*- coding: utf-8 -*-
"""
Created on Mon Mar 09 23:28:22 2015

@author: Greg
"""
import numpy as np
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict

class Activity():
    
    def get_acts(self,network):
        
        B = network.X.dot(network.Q)
        Th = np.tile(network.theta,(network.parameters.batch_size,1))
        Ys = np.zeros((network.parameters.batch_size,network.parameters.M))
        Y = np.zeros((network.parameters.batch_size,network.parameters.M))
        aas = np.zeros((network.parameters.batch_size,network.parameters.M))
        spike_train = np.zeros((network.parameters.batch_size,network.parameters.M,network.parameters.num_iterations))
        
        num_iterations = 50

        eta = .1
        
        for tt in xrange(num_iterations):
            Ys = (1.-eta)*Ys+eta*(B-aas.dot(network.W))
            aas = np.zeros((network.parameters.batch_size,network.parameters.M))
            #This resets the current activity of the time step to 0's        
            aas[Ys > Th] = 1.
            #If the activity of a given neuron is above the threshold, set it to 1 a.k.a. fire.
            
            spike_train[:,:,tt]=aas
            
            Y += aas
            Ys[Ys > Th] = 0.
            
        network.Y = Y
        network.spike_train = spike_train
        
class Activity_gpu():
    
    def __init__(self, network):
        X = network.X
        Q = network.Q
        theta = network.theta
        W = network.W
        Y = T.zeros_like(network.Y)
        Ys = T.zeros_like(Y)
        aas = T.zeros_like(Y)
        spike_train = network.spike_train
        Q_norms = (Q*Q).sum(axis=0, keepdims=True)

        B = X.dot(Q)
        Th = theta.dimshuffle('x', 0)

        num_iterations = network.parameters.num_iterations
        eta = .1

        for tt in xrange(num_iterations):
            Ys = (1.-eta*Q_norms)*Ys+eta*(B-aas.dot(W))
            aas = 0.*aas
            #This resets the current activity of the time step to 0's        
            aas = T.switch(Ys > Th, 1., aas)
            #If the activity of a given neuron is above the threshold, set it to 1 a.k.a. fire.
            
            spike_train = T.set_subtensor(spike_train[:,:,tt], aas)
            
            #Forces mean to be 0
            Y += aas
            #update total activity
            Ys = T.switch(Ys > Th, 0., Ys)

        updates = OrderedDict()
        updates[network.Y] = Y
        updates[network.spike_train] = spike_train
        self.f = theano.function([], [], updates=updates)
        
    def get_acts(self):
        self.f()
