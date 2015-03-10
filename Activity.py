# -*- coding: utf-8 -*-
"""
Created on Mon Mar 09 23:28:22 2015

@author: Greg
"""
import numpy as np
from Network import Network

class Activity():
    
    def get_acts(self,network):
        
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
        