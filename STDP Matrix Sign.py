# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:51:54 2015

@author: Bernal
"""

import numpy as np
import matplotlib.pyplot as plt

iterations = 50
time_dep= np.zeros((iterations,iterations))
post_activity=-2.7
pre_activity= 2.7 
time_scale=2
for i in xrange(iterations):
    for j in xrange(iterations):
        dt=i-j
        #i-j gives the correct signs to strengthen pre to post synaptic activity 10/05/14
        if np.sign(dt) == 1:
            time_dep[i][j]+= pre_activity*np.exp(-abs(dt*time_scale))*(dt)**16
        else:
            time_dep[i][j]+= post_activity*np.exp(-abs(dt*time_scale))*(dt)**16
            
#This is a test run with only two neurons that we will set to fire at different intervals and measure their inhibition

IPSP = np.array([])
delta_t = np.array([])

for i in xrange(iterations):
    spike_train = np.zeros((2,iterations))
    spike_train[0][i] = 1
    spike_train[1][iterations-i-1] = 1
    dW = np.dot(spike_train,np.dot(time_dep,spike_train.transpose()))
    IPSP = np.append(IPSP,dW[0][1])
    delta_t = np.append(delta_t,iterations-2*i-1)
    
plt.plot(delta_t,IPSP)