# -*- coding: utf-8 -*-
"""
Created on Mon Mar 09 22:30:12 2015

@author: Greg
"""
import ConfigParser
import numpy as np

class Network():
    
    def __init__(self,parameters_file):
        config = ConfigParser.ConfigParser()
        config.read(parameters_file)
        rng = np.random.RandomState(0)
        self.num_iterations = 50
        
        """
        Load network Parameters from config file
        """
        
        self.batch_size = config.getint("Parameters",'batch_size')
        self.num_trials = config.getint("Parameters",'num_trials')
        self.reduced_learning_rate = config.getfloat("Parameters",'reduced_learning_rate')
        self.N = config.getint("NeuronParameters",'N')
        self.OC = config.getint("NeuronParameters",'OC')
        self.p = config.getfloat("NeuronParameters",'p')
        self.alpha = config.getfloat("LearningRates",'alpha')
        self.beta = config.getfloat("LearningRates",'beta')
        self.gamma = config.getfloat("LearningRates",'gamma')
        self.eta_ave = config.getfloat("LearningRates",'eta_ave')
        self.lateral_constraint = config.getfloat('LearningRates','lateral_constraint')
        
        """
        Initialize X, W, Q, and theta; the input and network parameters.
        """        
        
        self.M = self.OC*self.N        
        self.Q = rng.randn(self.N,self.M)
        self.Q = self.Q.dot(np.diag(1./np.sqrt(np.diag(self.Q.T.dot(self.Q)))))
        self.W = np.zeros((self.M,self.M))
        self.theta = 2.*np.ones(self.M)
        self.X = np.zeros((self.batch_size,self.N))
        
        """
        These are used for the activities function
        """
        
        self.Y = np.zeros((self.batch_size,self.M))
        self.spike_train=np.zeros((self.batch_size,self.M,self.num_iterations))
        
    def ReduceLearning(self,tt):
        
        if tt >= 5000:
            self.gamma=self.gamma*self.reduced_learning_rate
            self.alpha=self.alpha*self.reduced_learning_rate
            self.beta=self.beta*self.reduced_learning_rate
        
    