# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 20:15:53 2014

@author: Greg Martin

Experiment used to determine the time dependent matrix for STDP
"""
import numpy as np
import time



batch_size=100
neurons=2048
iterations=50




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



a=np.random.randint(2,size=(batch_size,neurons,iterations))
print a.itemsize


stdp=np.zeros((neurons,neurons))
temp_time=time.time()

for i in xrange(batch_size):
    
    stdp+=np.dot(a[i],np.dot(time_dep,a[i].T))


dt=(time.time()-temp_time)/60
print "Done"






