# -*- coding: utf-8 -*-
"""
Created on Mon Mar 09 23:28:22 2015

@author: Greg
"""
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict
        
class Activity():
    
    def __init__(self, network):
        batch_size = network.parameters.batch_size
        num_iterations = network.parameters.num_iterations
        norm_infer = network.parameters.norm_infer
        X = network.X        
        updates = OrderedDict()

        for layer in range(network.n_layers):
            M = network.parameters.M[layer]
            Q = network.Q[layer]
            theta = network.theta[layer]
            W = network.W[layer]
            Y = T.alloc(0.,batch_size,M)[layer]
            Ys = T.zeros_like(Y)[layer]
            aas = T.zeros_like(Y)[layer]
            keep_spikes = False
            if hasattr(network, 'spike_train'):
                keep_spikes = True
                spike_train = T.alloc(0.,batch_size,M,num_iterations)[layer]
            
            Q_norm = (Q*Q).sum(axis=0, keepdims=True)
    
            B = X.dot(Q)
            Th = theta.dimshuffle('x', 0)
    
            eta = .1
    
            for tt in xrange(num_iterations):
                if norm_infer:
                    Ys = (1.-eta*Q_norm)*Ys+eta*(B-aas.dot(W))
                else:
                    Ys = (1.-eta)*Ys+eta*(B-aas.dot(W))
                aas = 0.*aas
                #This resets the current activity of the time step to 0's        
                aas = T.switch(Ys > Th, 1., aas)
                #If the activity of a given neuron is above the threshold, set it to 1 a.k.a. fire.
                
                if keep_spikes:
                    spike_train = T.set_subtensor(spike_train[:,:,tt], aas)
                
                #Forces mean to be 0
                Y += aas
                #update total activity
                Ys = T.switch(Ys > Th, 0., Ys)
            
            #Setting input of next layer to spikes of current one
            X = Y
            updates[network.Y[layer]] = Y
            
            if keep_spikes:
                updates[network.spike_train] = spike_train
        
        self.f = theano.function([], [], updates=updates)
        
    def get_acts(self):
        self.f()
        
