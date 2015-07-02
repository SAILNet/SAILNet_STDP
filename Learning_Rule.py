# -*- coding: utf-8 -*-
"""
Created on Mon Mar 09 22:22:28 2015

@author: Greg
"""

import numpy as np
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict

"Base Class for Implementing Learning Rules"
class Learning_Rule(object):
    
 
    
    def CalculateChange(self):
        raise NotImplementedError
    
    def Update(self):
        raise NotImplementedError

    def ReduceLearning(self,tt):
        network = self.network
        if tt <= 5000:
            network.parameters.gamma.set_value(network.parameters.gamma.get_value()*network.parameters.reduced_learning_rate)
            network.parameters.beta.set_value(network.parameters.beta.get_value()*network.parameters.reduced_learning_rate)
            network.parameters.alpha.set_value(network.parameters.alpha.get_value()*network.parameters.reduced_learning_rate)
            


"Classic SAILNet Learning Rule"
class SAILNet_rule(Learning_Rule):
    
    
    def CalculateChange(self,network):
        """        
        Calculate change in Lateral Weights dW
        """
        self.dW = network.alpha*(network.Cyy-network.p**2)
        
        
        """
        Calculate Change in Feed-Forward Weights dQ
        """
        square_act = np.sum(network.Y*network.Y,axis=0)
        mymat = np.diag(square_act)
        self.dQ = network.beta*network.X.T.dot(network.Y)/network.batch_size - network.beta*network.Q.dot(mymat)/network.batch_size
                
        """
        Calculate Change in Threshold Weights dtheta
        """        
        self.dtheta = network.gamma*(np.sum(network.Y,axis=0)/network.batch_size-network.p)
        
    def Update(self, network):
        self.CalculateChange(network)        
        
        network.W += self.dW
        network.W = network.W-np.diag(np.diag(network.W))
        network.W[network.W < 0] = 0.        
        
        
        network.Q += self.dQ
        network.theta += self.dtheta
        
"Classic SAILNet Learning Rule (Theano Version)"
class SAILNet_rule_gpu(Learning_Rule):
    
    def __init__(self, network):
        self.network = network
        self.parameters = network.parameters
        Y = network.Y
        X = network.X
        Q = network.Q
        W = network.W
        theta = network.theta
        p = self.parameters.p
        alpha = self.parameters.alpha
        beta = self.parameters.beta
        gamma = self.parameters.gamma
        batch_size = self.parameters.batch_size

        """        
        Calculate change in Lateral Weights dW
        """
        Cyy = Y.T.dot(Y)/batch_size
        muy = Y.mean(axis=0)
        
        dW = alpha*(Cyy - p**2)
        dW=dW.astype('float32')
        W = W+dW
        W = W - T.diag(T.diag(W))
        W = T.switch(W < 0.,0.,W)
        
        """
        Calculate Change in Feed-Forward Weights dQ
        """        
        square_act = T.sum(Y*Y,axis=0)
        mymat = T.diag(square_act)
        dQ = beta*(X.T.dot(Y) - (Q.dot(mymat)))/batch_size        
        Q = Q+dQ

        
        """
        Calculate Change in Threshold Weights dtheta
        """        
        dtheta = gamma*(muy - p)
        theta = (theta+dtheta).astype('float32')

        updates = OrderedDict()
        updates[network.Q] =Q
        updates[network.W] = W
        updates[network.theta] = theta
        
        self.f = theano.function([], [], updates=updates)
        
    def Update(self):
        self.f()
        
"STDP Learning Rule Based on Haas 2006 Paper"
class Exp_STDP(Learning_Rule):
    
    def __init__(self, model):
        self.model = model
        """
        This is the newer model of STDP based on an experimental fit.
        """
        if model == "New":
            iterations = 50
            self.time_dep= np.zeros((iterations,iterations))
            post_activity=-2.7
            pre_activity= 27 
            time_scale=2
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
                        
    
    def CalculateChange(self,network):
        """        
        Calculate change in Lateral Weights dW
        """
        self.dW=np.zeros((network.M,network.M))
        
        for batch in xrange(network.batch_size):
            self.dW+=np.dot(network.spike_train[batch],np.dot(self.time_dep,network.spike_train[batch].T))
        self.dW = self.dW/network.batch_size
            
        
        """
        Calculate Change in Feed-Forward Weights dQ
        """
        square_act = np.sum(network.Y*network.Y,axis=0)
        mymat = np.diag(square_act)
        self.dQ = network.beta*network.X.T.dot(network.Y)/network.batch_size - network.beta*network.Q.dot(mymat)/network.batch_size
                
        """
        Calculate Change in Threshold Weights dtheta
        """        
        self.dtheta = network.gamma*(np.sum(network.Y,axis=0)/network.batch_size-network.p)
        
    def Update(self, network):
        self.CalculateChange(network)
        network.W += self.dW
        network.W -= network.lateral_constraint*network.W
        network.W = network.W-np.diag(np.diag(network.W))
        network.W[network.W < 0] = 0.        
        
        
        network.Q += self.dQ
        network.theta += self.dtheta
        
    def polarityTest(self, network):
        
        spikeTrain = np.zeros([network.M, 50])
        spikeTrain[0][0] = 1
        spikeTrain[10][1] = 1
        
        dw = np.dot(spikeTrain,np.dot(self.time_dep,spikeTrain.T))
        
        if dw[0][10] < dw[10][0]:
            return True
                
        else:
            return False
        

"STDP based learning rule using Theano"

class Exp_STDP_gpu(Learning_Rule):
    
    def __init__(self,network):
        self.network = network
        self.parameters = network.parameters
        self.CreateMatrix()
        Y = network.Y
        X = network.X
        Q = network.Q
        W = network.W
        spike_train = network.spike_train
        theta = network.theta
        p = self.parameters.p
        beta = self.parameters.beta
        gamma = self.parameters.gamma
        batch_size = self.parameters.batch_size
        
        """
        Calculate Change in Feed-Forward Weights dW
        """
        dW=T.zeros_like(W).astype('float32')
        
        for batch in xrange(batch_size):
            
            dW = dW + T.dot(spike_train[batch], T.dot(self.time_dep,T.transpose(spike_train[batch])))
        dW = dW/batch_size
        
        
        W = W + dW
        W = W - T.diag(T.diag(W))
        W = T.switch(T.lt(W,T.zeros_like(W)),0.,W)
        
        """
        Calculate Change in Feed-Forward Weights dQ
        """        
        square_act = T.sum(Y*Y,axis=0)
        mymat = T.diag(square_act)
        dQ = beta*(T.dot(T.transpose(X),Y))/batch_size - beta*(T.dot(Q,mymat))/batch_size        
        Q = Q+dQ

        
        """
        Calculate Change in Threshold Weights dtheta
        """        
        dtheta = gamma*(T.sum(Y,axis = 0)/batch_size - p)
        theta = (theta+dtheta).astype('float32')
        
        updates = OrderedDict()
        updates[network.Q] = Q
        updates[network.W] = W
        updates[network.theta] = theta

        self.f = theano.function([], [dW], updates=updates)
                        
    def CreateMatrix(self):
        iterations = 50
        self.time_dep= np.zeros((iterations,iterations))
        post_activity=-2.7
        pre_activity= 27 
        time_scale=2
        for i in xrange(iterations):
            for j in xrange(iterations):
                
                dt=i-j
                #i-j gives the correct signs to strengthen pre to post synaptic activity 10/05/14
                if np.sign(dt) == 1:
                    self.time_dep[i][j]+= pre_activity*np.exp(-abs(dt*time_scale))*(dt)**16
                else:
                    self.time_dep[i][j]+= post_activity*np.exp(-abs(dt*time_scale))*(dt)**16
                    
        self.time_dep = theano.shared(self.time_dep.astype('float32'))
        
    def Update(self):
        self.dW = self.f()
        
    def polarityTest(self, network):
        
        spikeTrain = np.zeros([self.parameters.M, 50])
        spikeTrain[0][0] = 1
        spikeTrain[10][1] = 1
        
        dw = np.dot(spikeTrain,np.dot(self.time_dep,spikeTrain.T))
        
        if dw[0][10] < dw[10][0]:
            return True
                
        else:
            return False
        
    
        
        
    
