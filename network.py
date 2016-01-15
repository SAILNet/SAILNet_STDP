# -*- coding: utf-8 -*-
"""
Created on Mon Mar 09 22:30:12 2015

@author: Greg
"""
import numpy as np
import theano

shared_type = theano.tensor.sharedvar.SharedVariable

def make_shared(shape_or_array, val=0.):
    if isinstance(shape_or_array, (tuple, int)):
        return theano.shared(np.array(val, dtype='float32') *
                             np.ones(shape_or_array).astype('float32'))
    elif isinstance(shape_or_array, np.ndarray):
        return theano.shared(shape_or_array.astype('float32'))
    else:
        raise ValueError

class Network():
    
    def __init__(self, parameters):
        
        self.parameters = parameters
        time_data = parameters.time_data or parameters.movie_data
        keep_spikes = parameters.keep_spikes
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
        if time_data:
            self.X_tm1 = make_shared((parameters.batch_size, parameters.N))
            self.aas_tm1 = ()
            self.Ys_tm1 = ()
        if keep_spikes:
            self.spike_train = ()
            if time_data:
                self.spike_train_tm1 = ()

        self.X = make_shared((parameters.batch_size, parameters.N))

        nin = (parameters.N,)+parameters.M
        nout = parameters.M

        for ii in range(self.n_layers):
            in_dim = nin[ii]
            out_dim = nout[ii]
            Q = rng.randn(in_dim,out_dim)
            Q = 0.5*Q.dot(np.diag(1./np.sqrt(np.diag(Q.T.dot(Q)))))
            self.Q += (make_shared(Q),)
            self.W += (make_shared((out_dim, out_dim)),)
            self.theta += (make_shared(out_dim, val=.5),)
        
            """
            Save Spikes per Trial and Spike History as Theano Shared Variables
            """
            
            self.Y += (make_shared((parameters.batch_size, out_dim)),)
            if time_data:
                self.aas_tm1 += (make_shared((parameters.batch_size, out_dim)),)
                self.Ys_tm1 += (make_shared((parameters.batch_size, out_dim)),)
            if keep_spikes:
                self.spike_train += (make_shared((parameters.batch_size,
                                                  out_dim,
                                                  parameters.num_iterations)),)
                if time_data:
                    self.spike_train_tm1 += (make_shared((parameters.batch_size,
                                                          out_dim,
                                                          parameters.num_iterations)),)
        if keep_spikes:
            self.time_dep = make_shared((parameters.num_iterations,
                                         parameters.num_iterations))

    def initialize_time(self):
        if self.parameters.time_data or self.parameters.movie_data:
            for layer in range(self.n_layers):
                Ys_tm1 = self.Ys_tm1[layer]
                spike_train_tm1 = self.spike_train_tm1[layer]
                Ys_tm1.set_value(0.*Ys_tm1.get_value())
                spike_train_tm1.set_value(0.*spike_train_tm1.get_value())
        else:
            raise ValueError

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
                updates[key] = make_shared(value)
            elif isinstance(value, tuple) and all(isinstance(v, np.ndarray) for v in value):
                updates[key] = tuple(make_shared(v) for v in value)
        self.__dict__.update(updates)
