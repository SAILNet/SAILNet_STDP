# -*- coding: utf-8 -*-
"""
Created on Mon Mar 09 22:30:12 2015

@author: Greg
"""
import numpy as np
import theano

shared_type = theano.tensor.sharedvar.SharedVariable

class Network():
    
    def __init__(self, parameters):
        
        self.parameters = parameters
        rng = np.random.RandomState(1246)
        self.current_trial = 0
        self.total_trials = 0
        self.n_layers = parameters.n_layers

        """
        Initialize X, W, Q, and theta as Theano Shared Variables
        """        
        self.Q = ()
        self.W = ()
        self.theta = ()
        self.Y = ()
        if parameters.keep_spikes:
            self.spike_train = ()
            self.temp_dep = ()
        self.X = theano.shared(np.zeros((parameters.batch_size,parameters.N)).astype('float32'))

        nin = (parameters.N,)+parameters.M
        nout = parameters.M

        for ii in self.n_layers:
            in_dim = nin[ii]
            out_dim = nout[ii]
            Q = rng.randn(in_dim,out_dim)
            Q = 0.5*Q.dot(np.diag(1./np.sqrt(np.diag(Q.T.dot(Q)))))
            self.Q += (theano.shared(Q.astype('float32')),)
            self.W += (theano.shared(np.zeros((out_dim, out_dim)).astype('float32')),)
            self.theta = (theano.shared(0.5*np.ones(out_dim).astype('float32')),)
        
            """
            Save Spikes per Trial and Spike History as Theano Shared Variables
            """
            
            self.Y += (theano.shared(np.zeros((parameters.batch_size,out_dim)).astype('float32')),)
            if parameters.keep_spikes:
                self.spike_train += (theano.shared(np.zeros((parameters.batch_size,
                                                           out_dim,
                                                           parameters.num_iterations)).astype('float32')),)
                self.time_dep += (theano.shared(np.zeros((parameters.num_iterations
                                                          parameters.num_iterations)).astype('float32')),)

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
            if isinstance(value, shared_type):
                updates[key] = value.get_value()
            elif isinstance(value, tuple) and all(isinstance(v, shared_type) for v in value):
                updates[key] = tuple(v.get_value() for v in value)
        self.__dict__.update(updates)

    def to_gpu(self):
        items = self.__dict__
        updates = {}
        for key, value in items.iteritems():
            if isinstance(value, np.ndarray):
                updates[key] = theano.shared(value.astype('float32'))
            elif isinstance(value, tuple) and all(isinstance(v, np.ndarray) for v in value):
                updates[key] = tuple(theano.shared(v.astype('float32')) for v in value)
        self.__dict__.update(updates)
