import numpy as np
import cPickle, time
from math import ceil


def activities(X,Q,W,theta):
    batch_size, N = X.shape
    sz = int(np.sqrt(N))

    M = Q.shape[1]
    
    """
    Q is the matrix of connection strengths from each input to each neuron. it is a inputs X number of neurons
    """

    num_iterations = 50

    eta = .1

    B = X.dot(Q)
    #weighting the input activity by the feed-forward weights

    T = np.tile(theta,(batch_size,1))

    Ys = np.zeros((batch_size,M))
    aas = np.zeros((batch_size,M))
    Y = np.zeros((batch_size,M))
    stdp=np.zeros((batch_size,M,num_iterations))
    
    
    """    
    aas determines who spikes. Subtracting aas.dot(W) creates inhibition based on the weight.
    aas is either 1 or 0, either fired or not.

    (1 - eta)*Ys is a decay term.
    
    eta*(B) is a term that increases the activity based on the strength of the input
    weighted by the feed forward weights.
    
    eta*aas.dot(W) term is the inhibitory term.    
    """
    for tt in xrange(num_iterations):
        Ys = (1.-eta)*Ys+eta*(B-aas.dot(W))
        aas = np.zeros((batch_size,M))
        #This resets the current activity of the time step to 0's        
        aas[Ys > T] = 1.
        #If the activity of a given neuron is above the threshold, set it to 1 a.k.a. fire.
        
        
        """        
        Second attempt at STDP, using more matricies     
        """
        stdp[:,:,tt]=aas
        
        
        
        
        Y += aas
        #update total activity
        Ys[Ys > T] = 0.
        #after firing set back to zero for activity calculations in next time step
    
        
    return [Y,stdp]
<<<<<<< HEAD

=======
    
def STDP(M,model,iterations):
    
    
    time_dep= np.zeros((iterations,iterations))

    if model == "New":
       post_activity=-.070
       pre_activity=.070
       time_scale=4
       for i in xrange(iterations):
            for j in xrange(iterations):
                
                dt=j-i
                #j-i gives the correct signs to strengthen pre to post synaptic activity
                if np.sign(dt) == 1:
                    time_dep[i][j]+= pre_activity*np.exp(-abs(dt*time_scale))*(dt)**16
                else:
                    time_dep[i][j]+= post_activity*np.exp(-abs(dt*time_scale))*(dt)**16
                
       
    
    else:
        #09/17/14 Determined that post_activity=-10 pre_activity=5 and time scale=2 
        #makes the norm of the stdp array much smaller than that of dW
        post_activity=-45
        pre_activity=25
        time_scale=1
        for i in xrange(iterations):
            for j in xrange(iterations):
                if i !=j:
                    dt=j-i
                    #j-i gives the correct signs to strengthen pre to post synaptic activity
                    if np.sign(dt) == 1:
                        time_dep[i][j]+= pre_activity*np.exp(-abs(dt/time_scale))
                    else:
                        time_dep[i][j]+= post_activity*np.exp(-abs(dt/time_scale))
                else:
                    time_dep[i][j]=0
    
 
                
    return time_dep
>>>>>>> newstdp
rng = np.random.RandomState(0)

# Parameters
batch_size = 50
num_trials = 1000

# Load Images
with open('images.pkl','r') as f:
    images = cPickle.load(f)
imsize, imsize, num_images = images.shape
images = np.transpose(images,axes=(2,0,1))

BUFF = 20

# Neuron Parameters
N = 256
sz = np.sqrt(N).astype(np.int)
OC = 2 #Over-Completeness: num of neurons = OC * num of inputs
M = OC*N #M is the number of neurons

# Network Parameters
p = .05 #Sparcity

# Initialize Weights
Q = rng.randn(N,M)
Q = Q.dot(np.diag(1./np.sqrt(np.diag(Q.T.dot(Q)))))
#1./np.sqrt(np.diag(Q.T.dot(Q))) normalizes the Q matrix
W = np.zeros((M,M))
theta = 2.*np.ones(M)

# Learning Rates
alpha = 1.
beta = .01
gamma = .1

eta_ave = .3

Y_ave = p
Cyy_ave = p**2


Cyy_ave_pertrial=np.zeros(num_trials)

# Zero timing variables
data_time = 0.
algo_time = 0.


"""
This will create a matrix of weights for various positions in time.
The iterations variable needs to be the same as in the activity function. Plan
to pass this variable into the activity function later.

The weights are determined by an exponential.

The post and pre activity weights represent the bias towards connection
strength weakening observed in actual STDP.
"""
time_for_stdp=time.time()
stdp=np.zeros((M,M))
iterations=50
time_dep= np.zeros((iterations,iterations))


#09/17/14 Determined that post_activity=-10 pre_activity=5 and time scale=2 
#makes the norm of the stdp array much smaller than that of dW
post_activity=-45
pre_activity=25
time_scale=1
for i in xrange(iterations):
    for j in xrange(iterations):
        if i !=j:
            dt=j-i
            #j-i gives the correct signs to strengthen pre to post synaptic activity
            if np.sign(dt) == 1:
                time_dep[i][j]+= pre_activity*np.exp(-abs(dt/time_scale))
            else:
                time_dep[i][j]+= post_activity*np.exp(-abs(dt/time_scale))
        else:
            time_dep[i][j]=0

time_for_stdp= time.time()-time_for_stdp

#The following will keep track of the change of the magnitude of the stdp
#matrix for each trial.

mag_stdp=np.zeros(num_trials)

#mag_dW will track the magnitude changes in dW

mag_dW=np.zeros_like(mag_stdp)

#Correlation matrix for each neuron

cor_dW_stdp=np.zeros_like(mag_stdp)



# Begin Learning
X = np.zeros((batch_size,N))

for tt in xrange(num_trials):
    # Extract image patches from images
    dt = time.time()
    for ii in xrange(batch_size):
        r = BUFF+int((imsize-sz-2.*BUFF)*rng.rand())
        c = BUFF+int((imsize-sz-2.*BUFF)*rng.rand())
        myimage = images[int(num_images*rng.rand()),r:r+sz,c:c+sz].ravel()
        #takes a chunck from a random image, size of 16X16 patch at a random location       
        myimage = myimage-np.mean(myimage)
        myimage = myimage/np.std(myimage)
        #Forces mean to be 0
        X[ii] = myimage
        #creating a list of image patches to work with
        
    dt = time.time()-dt
    data_time += dt/60.
    

    dt = time.time()
    # Calcuate network activities
    Y, activity_log = activities(X,Q,W,theta)
   
    """
    This commented out section was used to determine the sign for time_dep
    activity_log=np.zeros((batch_size,M,iterations))
    activity_log[0][0][0]+=1
    activity_log[0][10][1]+=1
    """
    muy = np.mean(Y,axis=0)
    Cyy = Y.T.dot(Y)/batch_size
    
    """
    using stdp matrix to update W
    """
    
    time_stdp=time.time()
    
    for batch in xrange(batch_size):
        stdp+=np.dot(activity_log[batch],np.dot(time_dep,activity_log[batch].T))
    stdp = stdp/batch_size
    time_stdp= time.time()-time_stdp
    time_for_stdp+= time_stdp
    
    mag_stdp[tt]=np.linalg.norm(stdp)
    
    """
    The following code is the learning rules
    """    
    
    # Update lateral weigts
    dW = alpha*(Cyy-p**2)
    W += dW
    W = W-np.diag(np.diag(W))
    W[W < 0] = 0.
    
    mag_dW[tt]=np.linalg.norm(dW)

    # Update feedforward weights
    square_act = np.sum(Y*Y,axis=0)
    mymat = np.diag(square_act)
    dQ = beta*X.T.dot(Y)/batch_size - beta*Q.dot(mymat)/batch_size
    Q += dQ

    # Update thresholds
    dtheta = gamma*(np.sum(Y,axis=0)/batch_size-p)
    theta += dtheta
    dt = time.time()-dt
    algo_time += dt/60.
    time_for_stdp= time_for_stdp/60
    
    """
    We shall determine the correlation between dW and stdp by dW*stdp/(|dW||stdp|)
    """
    cor_dW_stdp[tt]=sum(sum(dW.dot(stdp)))/(np.linalg.norm(dW)*np.linalg.norm(stdp))
    
    Y_ave = (1.-eta_ave)*Y_ave + eta_ave*muy
    Cyy_ave=(1.-eta_ave)*Cyy_ave + eta_ave*Cyy
    Cyy_ave_pertrial[tt]=sum(sum(Cyy))
    
    if tt%50 == 0 and tt != 0:
        print 'Batch: '+str(tt)+' out of '+str(num_trials)
        print 'Cumulative time spent gathering data: '+str(data_time)+' min'
        print 'Cumulative time spent in SAILnet: '+str(algo_time)+' min'
        print 'Cumulative time spent calculating STDP weights: '+str(time_for_stdp)+' min'
        print ''
    total_time = data_time+algo_time+time_for_stdp
print 'Percent time spent gathering data: '+str(data_time/total_time)+' %'
print 'Percent time spent in SAILnet: '+str(algo_time/total_time)+' %'
print 'Percent time spent calculating STDP: '+str(time_for_stdp/total_time)+' %'
print '' 
 

with open('Plotting\dW' + str(num_trials)+'.pkl','wb') as f:
    cPickle.dump((W,Q,theta,stdp,mag_stdp,mag_dW,cor_dW_stdp,Y_ave,Cyy_ave_pertrial),f)


