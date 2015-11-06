# -*- coding: utf-8 -*-
"""
Created on Mon Mar 09 23:28:22 2015

@author: Greg
"""
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict

class BaseActivity(object):
    def get_acts(self):
        raise NotImplementedError
        
class Activity(BaseActivity):
    
    def __init__(self, network):
        self.trial_num = 0
        batch_size = network.parameters.batch_size
        num_iterations = network.parameters.num_iterations
        keep_spikes = network.parameters.keep_spikes
        norm_infer = network.parameters.norm_infer
        if hasattr(network.parameters, 'firing_decay'):
            firing_decay = network.parameters.firing_decay
        else:
            firing_decay = False
        #firing_decay = False
        time_data = network.parameters.time_data
        X = network.X
        updates = OrderedDict()
        for layer in range(network.n_layers):
            M = network.parameters.M[layer]
            Q = network.Q[layer]
            theta = network.theta[layer]
            W = network.W[layer]
            Y = T.alloc(0., batch_size, M)
            if time_data and self.trial_num != 0:
                Ys = network.Ys_tm1[layer]
                aas = network.aas_tm1[layer]
            else:
                Ys = T.zeros_like(Y)
                aas = T.zeros_like(Y)
            if keep_spikes:
                spike_train = T.alloc(0., batch_size, M, num_iterations)
            
            Q_norm = (Q*Q).sum(axis=0, keepdims=True)
    
            B = X.dot(Q)
            Th = theta.dimshuffle('x', 0)
    
            eta = .1
    
            for tt in xrange(num_iterations):
                if norm_infer:
                    Ys = (1.-eta*Q_norm)*Ys+eta*(B-aas.dot(W))
                elif firing_decay:
                    Ys = (1.-eta)*Ys+eta*(B-Y.dot(W))
                else:
                    Ys = (1.-eta)*Ys+eta*(B-aas.dot(W))

                aas = 0.*aas
                # This resets the current activity of the time step to 0's        
                aas = T.switch(Ys > Th, 1., aas)

                # If the activity of a given neuron is above the threshold, set it to 1 a.k.a. fire.
                
                if keep_spikes:
                    spike_train = T.set_subtensor(spike_train[:,:,tt], aas)
                
                Y += aas
                # Update total activity
                Ys = T.switch(Ys > Th, 0., Ys)
            
            # Setting input of next layer to spikes of current one
            X = Y
            updates[network.Y[layer]] = Y
            
            if keep_spikes:
                if time_data:
                    updates[network.spike_train_tm1[layer]] = network.spike_train[layer]
                updates[network.spike_train[layer]] = spike_train
            if time_data:
                updates[network.Ys_tm1[layer]] = Ys
                updates[network.aas_tm1[layer]] = aas
        
        self.f = theano.function([], [], updates=updates)
        
    def get_acts(self):
        self.trial_num += 1
        self.f()
