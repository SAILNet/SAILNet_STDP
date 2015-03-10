# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 15:51:52 2015

@author: Bernal
"""

import numpy as np
from Learning_Rule import Learning_Rule


class STDP_Rule(Learning_Rule):
    
    def __init__(self, model, network):
        self.model = model
        """
        This is the newer model of STDP based on an experimental fit.
        """
        if model == "New":
            iterations = 50
            self.time_dep= np.zeros((iterations,iterations))
            post_activity=-.027
            pre_activity=.027 #This one needs to be negative
            time_scale=4
            for i in xrange(iterations):
                for j in xrange(iterations):
                    
                    dt=i-j
                    #i-j gives the correct signs to strengthen pre to post synaptic activity 10/05/14
                    if np.sign(dt) == 1:
                        self.time_dep[i][j]+= pre_activity*np.exp(-abs(dt*time_scale))*(dt)**16
                    else:
                        self.time_dep[i][j]+= post_activity*np.exp(-abs(dt*time_scale))*(dt)**16
        
        """
        This is the first model of STDP based on an experimental fit.
        """
        if model == "Old":
            iterations = 50
            self.time_dep= np.zeros((iterations,iterations))
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
                            self.time_dep[i][j]+= pre_activity*np.exp(-abs(dt/time_scale))
                        else:
                            self.time_dep[i][j]+= post_activity*np.exp(-abs(dt/time_scale))
                    else:
                        self.time_dep[i][j]=0
                        
    
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
    
    def Update(self,dW,W,stdp,time_dep,activity_log,batch_size,tt):
        """
        STDP matrix to update W(lateral connection strengths)
        """
               
        for batch in xrange(batch_size):
            stdp+=np.dot(activity_log[batch],np.dot(self.time_dep,activity_log[batch].T))
        stdp = stdp/batch_size
            
        self.mag_stdp[tt]=np.linalg.norm(stdp)
        W += stdp
        W = W-np.diag(np.diag(W))
        W[W < 0] = 0.
        
        self.mag_dW[tt]=np.linalg.norm(dW)
        self.mag_W[tt] =np.linalg.norm(W)
        
        return W, stdp
        
    
        