# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 00:01:16 2015

@author: Greg
"""

import numpy as np
from Learning_Rule import Learning_Rule


class SAILNet_rule(Learning_Rule):
    def __init__(self):
        pass
        
    
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
        
        network.W += self.dW
        network.W = network.W-np.diag(np.diag(network.W))
        network.W[network.W < 0] = 0.        
        
        
        network.Q += self.dQ
        network.theta += self.dtheta