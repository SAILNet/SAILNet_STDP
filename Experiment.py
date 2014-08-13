# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 20:15:53 2014

@author: Greg Martin
"""
import numpy as np



batch_size=10
neurons=16
iterations=5




time_dep= np.zeros((iterations,iterations))
for i in xrange(iterations):
    for j in xrange(iterations):
        if i !=j:
            time_dep[i][j]+= 2/float(i-j)**3
        else:
            time_dep[i][j]=0



#The above just creates a weighting matrix for the tensordot approach

"""
I think this does it. I tested with an array which I know that neurons 1, 2, and 5 are highly correlated.
The results are as expected.
"""



a=np.array([[[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,0,0,0],[0,1,0,0,0]],
            [[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,0,0,0],[0,1,0,0,0]],
            [[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,0,0,0],[0,1,0,0,0]],
            [[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,0,0,0],[0,1,0,0,0]],
            [[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,0,0,0],[0,1,0,0,0]]])

stdp=np.dot(a,np.dot(time_dep,a.transpose(1,2,0)).transpose(2,0,1)).sum(0).sum(1)






