# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 15:51:52 2015

@author: Bernal
"""

import numpy as np
import cPickle, time
from math import ceil
from pca import pca
import van_hateren as VH
from utils import tile_raster_images
import matplotlib.pyplot as plt
import ConfigParser
import os
import shutil
from SAILnet_Plotting import Plot

class STDP_Rule():
    
    def __init__(self, model):
        self.model = model
        
    def activities(self,X,Q,W,theta):
        batch_size, N = X.shape
        sz = int(np.sqrt(N))
    
        M = Q.shape[1]
        
        """
        Q is the matrix of connection strengths from each input to each neuron. it is an (Inputs X number of neurons) matrix
        """
    
        num_iterations = 50
    
        eta = .1
    
        B = X.dot(Q)
        #weighting the input activity by the feed-forward weights
    
        T = np.tile(theta,(batch_size,1))
    
        Ys = np.zeros((batch_size,M))
        aas = np.zeros((batch_size,M))
        Y = np.zeros((batch_size,M))
        stdp=np.zeros((batch_size,M,num_iterations))
        
        
        """    
        aas determines who spikes. Subtracting aas.dot(W) creates inhibition based on the weight.
        aas is either 1 or 0, either fired or not.
    
        (1 - eta)*Ys is a decay term.
        
        eta*(B) is a term that increases the activity based on the strength of the input
        weighted by the feed forward weights.
        
        eta*aas.dot(W) term is the inhibitory term.    
        """
        for tt in xrange(num_iterations):
            Ys = (1.-eta)*Ys+eta*(B-aas.dot(W))
            aas = np.zeros((batch_size,M))
            #This resets the current activity of the time step to 0's        
            aas[Ys > T] = 1.
            #If the activity of a given neuron is above the threshold, set it to 1 a.k.a. fire.
            
            
            """        
            Second attempt at STDP, using more matricies     
            """
            stdp[:,:,tt]=aas
            
            
            
            #Forces mean to be 0
            Y += aas
            #update total activity
            Ys[Ys > T] = 0.
            #after firing set back to zero for activity calculations in next time step
        
            
        return [Y,stdp]
        
    
    def STDP_Matrix(self,M,iterations):
        
        time_dep= np.zeros((iterations,iterations))
    
        if self.model == "New":
           post_activity=-.027
           pre_activity=.027 #This one needs to be negative
           time_scale=4
           for i in xrange(iterations):
                for j in xrange(iterations):
                    
                    dt=i-j
                    #i-j gives the correct signs to strengthen pre to post synaptic activity 10/05/14
                    if np.sign(dt) == 1:
                        time_dep[i][j]+= pre_activity*np.exp(-abs(dt*time_scale))*(dt)**16
                    else:
                        time_dep[i][j]+= post_activity*np.exp(-abs(dt*time_scale))*(dt)**16
        
        else:
            #09/17/14 Determined that post_activity=-10 pre_activity=5 and time scale=2 
            #makes the norm of the stdp array much smaller than that of dW
            post_activity=-45
            pre_activity=25
            time_scale=1
            for i in xrange(iterations):
                for j in xrange(iterations):
                    if i !=j:
                        dt=i-j
                        #i-j gives the correct signs to strengthen pre to post synaptic activity 10/05/14
                        if np.sign(dt) == 1:
                            time_dep[i][j]+= pre_activity*np.exp(-abs(dt/time_scale))
                        else:
                            time_dep[i][j]+= post_activity*np.exp(-abs(dt/time_scale))
                    else:
                        time_dep[i][j]=0
                        
        return time_dep
    
    def init_analysis(self, num_trials):
        
        #The following will keep track of the change of the magnitude of the stdp
        #matrix for each trial.        
        self.mag_stdp=np.zeros(num_trials)
        
        #mag_dW will track the magnitude changes in dW        
        self.mag_dW=np.zeros_like(self.mag_stdp)
        
        #mag_W will track the magnitude in W        
        self.mag_W = np.zeros_like(self.mag_stdp)
        
        #Correlation matrix for each neuron        
        self.cor_dW_stdp=np.zeros_like(self.mag_stdp)
        
        self.reconstruction_error=np.zeros_like(self.mag_dW)
    
    def STDP_analysis(self,dW,stdp,X,Y,Q,batch_size,N,tt):
    
        #We shall determine the correlation between dW and stdp by dW*stdp/(|dW||stdp|)
        self.cor_dW_stdp[tt]=sum(sum(dW.dot(stdp)))/(np.linalg.norm(dW)*np.linalg.norm(stdp))
    
        #Error in reconstucting the images
        self.reconstruction_error[tt]=np.sum(np.sum((X-Y.dot(Q.T))**2))/(2*N*batch_size)  
    
    def lateral_updates(self,dW,W,stdp,time_dep,activity_log,batch_size,tt):
        """
        STDP matrix to update W(lateral connection strengths)
        """
               
        for batch in xrange(batch_size):
            stdp+=np.dot(activity_log[batch],np.dot(time_dep,activity_log[batch].T))
        stdp = stdp/batch_size
            
        self.mag_stdp[tt]=np.linalg.norm(stdp)
        W += stdp
        W = W-np.diag(np.diag(W))
        W[W < 0] = 0.
        
        self.mag_dW[tt]=np.linalg.norm(dW)
        self.mag_W[tt] =np.linalg.norm(W)
        
        return W, stdp
        
    
        