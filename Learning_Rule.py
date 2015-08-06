# -*- coding: utf-8 -*-
"""
Created on Mon Mar 09 22:22:28 2015

@author: Greg
"""

import numpy as np
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict
    
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
            
class Learning_Rule_gpu(Learning_Rule):
    
    def __init__(self, network, dW_rule):
        self.network = network
        parameters = network.parameters
        Y = network.Y
        X = network.X
        Q = network.Q
        W = network.W
        theta = network.theta
        p = parameters.p
        beta = parameters.beta
        gamma = parameters.gamma
        batch_size = parameters.batch_size
        dW_Rule = str_to_dW[dW_rule](network)
        
        dW = dW_Rule.calc_dW()
	
	#dW,self.time_dep = dW_Rule.calc_dW()        
        
	mag_dW = T.sqrt(T.sum(T.sqr(dW)))

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
        muy = Y.mean(axis=0)
        dtheta = gamma*(muy - p)
        theta = (theta+dtheta).astype('float32')

        updates = OrderedDict()
        updates[network.Q] =Q
        updates[network.W] = W
        updates[network.theta] = theta
        
        self.f = theano.function([], [mag_dW], updates=updates)
        
    def Update(self):
        self.mag_dW = self.f()
        
"SAILNet Rule and Time Dependent Rules for dW"
        
class Abs_dW(object):
    
    def __init__(self,network):
        self.network = network

class dW_SAILnet(Abs_dW):
    
    def calc_dW(self):
        
        Y = self.network.Y
        alpha = self.network.parameters.alpha
        batch_size = self.network.parameters.batch_size
        p = self.network.parameters.p
        
        Cyy = Y.T.dot(Y)/batch_size
        dW = alpha*(Cyy - p**2)
        
        dW=dW.astype('float32')

        return dW
    
class dW_identity(Abs_dW):
    
    def calc_dW(self):
        spike_train = self.network.spike_train
        batch_size = self.network.parameters.batch_size
        num_iterations = self.network.parameters.num_iterations  
        p = self.network.parameters.p
        alpha = self.network.parameters.alpha        
        dW =  T.zeros_like(self.network.W).astype('float32')       
        
        min_constant = p**2/num_iterations
        dW = T.tensordot(spike_train, spike_train, axes=([0, 2], [0, 2]))
        #for batch in xrange(batch_size):
        #    dW = dW + T.dot(spike_train[batch], T.transpose(spike_train[batch]))
        
        dW = dW/batch_size
        dW = alpha*(dW - min_constant)
        
        return dW
        
class dW_time_dep(Abs_dW):
    
    def __init__(self,network):
        super(dW_time_dep,self).__init__(network)
        self.time_dep = time_matrix(str_to_fnc[network.parameters.function],self.network.parameters.num_iterations)
        
    def calc_dW(self):
        spike_train = self.network.spike_train
        batch_size = self.network.parameters.batch_size
        num_iterations = self.network.parameters.num_iterations  
        p = self.network.parameters.p
        alpha = self.network.parameters.alpha  
        dW =  T.zeros_like(self.network.W).astype('float32')
        
        P = p*np.ones(num_iterations,dtype= 'float32')
        min_constant = np.dot(P,np.dot(self.time_dep.get_value(),P))/num_iterations**2
        
        dW = T.tensordot(spike_train,self.time_dep,axes=([2],[0]))
        dW = T.tensordot(dW, spike_train,axes=([0,2],[0,2]))        
        #for batch in xrange(batch_size):
        #    dW = dW + T.dot(spike_train[batch], T.dot(self.time_dep,T.transpose(spike_train[batch])))
        
        dW = dW/batch_size  
        dW = alpha*(dW - min_constant)
        
        dW = dW.astype('float32')

        return dW#,self.time_dep
        
str_to_dW = {'dW_SAILnet': dW_SAILnet,
             'dW_identity': dW_identity,
             'dW_time_dep': dW_time_dep}
    
def time_matrix(function,iterations):
    
    time_dep= np.zeros((iterations,iterations))
    for i in xrange(iterations):
        for j in xrange(iterations):

            time_dep[i][j] = function(i,j)
                
    return theano.shared(time_dep.astype('float32'))  
        
        
"Time Dependent Functions"

def STDP(i,j):
    post_activity=-2.7
    pre_activity= 27 
    time_scale=2
    
    dt = i-j
    if np.sign(dt) == 1:
        return pre_activity*np.exp(-abs(dt*time_scale))*(dt)**16
    else:
        return post_activity*np.exp(-abs(dt*time_scale))*(dt)**16
    
def Unit(i,j): #Same as vanilla SAILNet
    return 1
    
def Step(i,j):
    dt = i-j
    step_len = 2
    step_height = 1
    if abs(dt) <= step_len:
        return step_height
    else:
        return 0
        
def Well(i,j):
    dt = i-j
    length = 2
    depth = 1
    if abs(dt) <= length/2:
        return 0
    else:
        return depth

def Gaussian(i,j):
    std = 5
    dt = i-j
    return np.exp(-0.5*(dt/std)**2)
    
    
str_to_fnc = {'STDP': STDP,
              'Unit': Unit,
              'Step': Step,
              'Well': Well,
              'Gaussian': Gaussian}        
        
        
        
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
        
    
