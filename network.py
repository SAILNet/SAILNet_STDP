# -*- coding: utf-8 -*-
"""
Created on Mon Mar 09 22:30:12 2015

@author: Greg
"""
import ConfigParser
import numpy as np
import theano

class Network():
    
    def __init__(self,parameters):
        
        self.parameters = parameters
        rng = np.random.RandomState(1246)
        self.current_trial = 0
        self.total_trials = 0
     
        """
        Initialize X, W, Q, and theta as Theano Shared Variables
        """        
        Q = rng.randn(parameters.N,parameters.M)
        self.Q = theano.shared(0.5*Q.dot(np.diag(1./np.sqrt(np.diag(Q.T.dot(Q))))).astype('float32'))
        self.W = theano.shared(np.zeros((parameters.M,parameters.M)).astype('float32'))
        self.theta = theano.shared(0.5*np.ones(parameters.M).astype('float32'))
        self.X = theano.shared(np.zeros((parameters.batch_size,parameters.N)).astype('float32'))
        
        """
        Save Spikes per Trial and Spike History as Theano Shared Variables
        """
        
        self.Y = theano.shared(np.zeros((parameters.batch_size,parameters.M)).astype('float32'))
        if parameters.keep_spikes:
            self.spike_train = theano.shared(np.zeros((parameters.batch_size,
                                                       parameters.M,
                                                       parameters.num_iterations)).astype('float32'))
        
    def continue_learning(self):
        if self.current_trial < self.parameters.num_trials:
            return True
        else:
            return False
            
    def next_trial(self):
        self.current_trial += 1
        self.total_trials  += 1
        
    def to_cpu(self):
        items = self.__dict__
        updates = {}
        for key, value in items.iteritems():
            if isinstance(value, theano.tensor.sharedvar.SharedVariable):
                updates[key] = value.get_value()
        self.__dict__.update(updates)

    def to_gpu(self):
        items = self.__dict__
        updates = {}
        for key, value in items.iteritems():
            if isinstance(value, np.ndarray):
                updates[key] = theano.shared(value.astype('float32'))
        self.__dict__.update(updates)
