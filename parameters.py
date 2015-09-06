# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:28:32 2015

@author: Bernal
"""
import ConfigParser
import numpy as np
import theano

spike_rules = ['dW_identity','dW_time_dep']

class Parameters():
    
    def __init__(self,parameters_file):
        config = ConfigParser.ConfigParser()
        config.read(parameters_file)
        
        """
        Load network Parameters from config file
        """
        
        self.dW_rule = config.get('LearningRule','dW_rule')
        self.update_keep_spikes()
        self.function = config.get('LearningRule','function')

        self.batch_size = config.getint("Parameters",'batch_size')
        self.num_images = config.getint("Parameters",'num_images')
        self.num_trials = config.getint("Parameters",'num_trials')
        self.num_iterations = config.getint("Parameters",'num_iterations')
        reduce_learning_rate = config.getfloat("Parameters",'reduce_learning_rate')
        self.reduce_learning_rate = np.array(reduce_learning_rate).astype('float32')
        self.norm_infer = config.getboolean("Parameters", "norm_infer")

        self.N = config.getint("NeuronParameters",'N')
        self.OC1 = config.getint("NeuronParameters",'OC1')
        self.OC2 = config.getint("NeuronParameters",'OC2')
        self.p = config.getfloat("NeuronParameters",'p')
        self.n_layers = config.getint("NeuronParameters",'n_layers')
        self.M = (self.N*self.OC1, self.N*self.OC2)

        self.alpha = theano.shared(np.array(config.getfloat("LearningRates",'alpha')).astype('float32'))
        self.beta = theano.shared(np.array(config.getfloat("LearningRates",'beta')).astype('float32'))
        self.gamma = theano.shared(np.array(config.getfloat("LearningRates",'gamma')).astype('float32'))        

    def update_keep_spikes(self):
        if self.dW_rule in spike_rules:
            self.keep_spikes = True
        else:
            self.keep_spikes = False
 
