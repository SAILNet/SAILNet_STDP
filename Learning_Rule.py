# -*- coding: utf-8 -*-
"""
Created on Mon Mar 09 22:22:28 2015

@author: Greg
"""

import numpy as np
import theano
import theano.tensor as T

"Base Class for Implementing Learning Rules"
class Learning_Rule(object):
    
 
    
    def CalculateChange(self):
        raise NotImplementedError
    
    def Update(self,network):
        self.CalculateChange(network)
        raise NotImplementedError


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
    
    def __init__(self):
        Cyy = T.matrix('Cyy')
        Y = T.matrix('Y')
        X = T.matrix('X')
        Q_o = T.matrix('Q')        
        alpha = T.scalar('alpha')
        p = T.scalar('p')                
        beta = T.scalar('beta')
        batch_size = T.scalar('batch_size')
        gamma = T.scalar('gamma')
        
        """        
        Calculate change in Lateral Weights dW
        """
        dW = alpha*(Cyy - p**2)
        
        """
        Calculate Change in Feed-Forward Weights dQ
        """        
        square_act = T.sum(Y*Y,axis=0)
        mymat = T.diag(square_act)
        dQ = beta*(T.dot(T.transpose(X),Y))/batch_size - beta*(T.dot(Q_o,mymat))/batch_size        
        
        """
        Calculate Change in Threshold Weights dtheta
        """        
        dtheta = gamma*(T.sum(Y,axis = 0)/batch_size - p)
        
        self.f_W = theano.function([alpha,p,Cyy],[dW])
        self.f_Q = theano.function([Y,X,Q_o,beta,batch_size],[dQ])
        self.f_T = theano.function([gamma,Y,batch_size,p],[dtheta])
        
    def CalculateChange(self,network):
        
        Cyy = network.Cyy.astype('float32')
        Y = network.Y.astype('float32')
        X = network.X.astype('float32')
        Q = network.Q.astype('float32')  
        alpha = network.alpha.astype('float32')
        p = network.p.astype('float32')                
        beta = network.beta.astype('float32')
        batch_size = network.batch_size.astype('float32')
        gamma = network.gamma.astype('float32')
        
        self.dW = self.f_W(alpha,p,Cyy)        
        self.dQ = self.f_Q(Y, X, Q, batch_size, beta)        
        self.dtheta = self.f_T(gamma,Y,batch_size,p)
        
    def Update(self, network):
        self.CalculateChange(network)
        
        W = T.matrix('W')
        
        W += self.dW
        W = W - T.diag(T.diag(W))
        W = T.switch(T.lt(W,T.zeros_like(W)),0,W)
        W_n = W
        
        f = theano.function([dW,W],[W_n])
        
        W = network.W.astype('float32')
        network.W = f(self.dW,W)
        
        network.Q += self.dQ
        network.theta += self.dtheta
        
        
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
        
        
        
    
        
        
    