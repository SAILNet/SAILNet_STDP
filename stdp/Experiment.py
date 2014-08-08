# -*- coding: utf-8 -*-
"""
Created on Tue Aug 05 13:15:14 2014

@author: Greg
"""

import numpy as np
import time

x= np.random.randint(2, size=(100,256,25))

y= np.random.randint(2, size=np.shape(x))
"""
print y

print np.transpose(x,(1,2,0)).shape



result = np.tensordot(np.transpose(y,(1,0,2)),np.transpose(x,(0,2,1)))/100

a=np.random.randint(2, size=(2,6,3))




print result
print result.shape

print np.transpose(a,(1,0,2))
print np.transpose(a,(0,2,1))
other= np.tensordot(np.transpose(a,(1,0,2)),np.transpose(a,(0,2,1)))
print other
print other.shape
"""

time_steps=50
batch_size=100
neurons=512

time_dep= np.zeros((time_steps,time_steps))

for i in xrange(time_steps):
    for j in xrange(time_steps):
        if i !=j:
            time_dep[i][j]+= 1/float((i-j))**3
        else:
            time_dep[i][j]=0
print time_dep



a=np.random.randint(2, size=(batch_size,neurons,time_steps))

temp_time=time.time()
try1=np.float32(np.tensordot(a.T,a,1).transpose(1,2,0,3))
dt=time.time()-temp_time
print try1.shape


outer=np.zeros_like(try1)


for l in xrange(neurons):
    for k in xrange(neurons):
        
        outer[l,k]=try1[i][j]*time_dep


one_outer=outer[0][0]



