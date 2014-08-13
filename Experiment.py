# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 20:15:53 2014

@author: Greg Martin
"""
import numpy as np



"""
result= np.dot(y.T,np.transpose(x,(1,0,2)))
print result
print np.shape(result)
"""
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


stdp=(np.tensordot(a.T,a,1).transpose((1,2,0,3))[:,:]*time_dep).sum(2).sum(2)

The idea of this function is that it creates a matrix for each possible combination of neurons
relating the time. It is like I took a vector for time of one neuron and dotted it with another
neuron's transpose of that vector, an outer product. The idea is that it would not be zero in
the time position where both of them fired. So [1][2] means the first neuron fired in time step
1 and the second in two. That position in the matrix would have a non-zero value which is then
weighted by the time_dep array. The matrix is then reduced by simply summing over the axes to
a single number which would then be stored in the matrix the size of W.

This idea is interesting but requires arrays far too large to reasonably use.
"""




"""
This was some an earlier attempt, before using the tensordot function. This has virtually the same functionality

for i in xrange(neuron):
    for j in xrange(neuron):
        for k in xrange(batch_size):
            for l in xrange(batch_size):
                stdp[i][j]=(np.outer(a[k][i],a[l][j])*time_dep).sum(0).sum(0)

"""

#print stdp



a=np.array([[[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,0,0,0],[0,1,0,0,0]],
            [[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,0,0,0],[0,1,0,0,0]],
            [[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,0,0,0],[0,1,0,0,0]],
            [[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,0,0,0],[0,1,0,0,0]],
            [[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,0,0,0],[0,1,0,0,0]]])

stdp=np.dot(a,np.dot(time_dep,a.transpose(1,2,0)).transpose(2,0,1)).sum(0).sum(1)



"""
The above just makes a somewhat more sparse random array. Turns out to only be about twice as dense
as SAILNet.
"""

""""
This approach uses the nonzero function in hopes that the sparsity will keep the calculations down.
This turns out to be not true and the double for loop takes much too long. Interesting note however,
I decided to use the exponential function after reading a paper where a group used STDP in their
neural-network. The amount they changed W was similar to this.

stdp=np.zeros((neurons,neurons))
time_scale=1
neuron_time=np.asarray(a.nonzero())

Note: the nonzero function returns essentially an array with one axis being of length 3 and the other is the
length of the number of total firing. This means that the [0] corresponds to the batch number the activity
was in, [1] is which neuron, and [2] is which time step.

for i in range(len(neuron_time[1])):
    for j in range(len(neuron_time[1])):
        dt=neuron_time[2][i]-neuron_time[2][j]
        stdp[neuron_time[1][i],neuron_time[1][j]]+=np.sign(dt)*np.exp(-np.abs(dt)/time_scale)

"""



