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
    STDP = np.zeros(W.shape)
    
    stdp_change=.1
    
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
        The following is Greg's first attempt at implementing STDP. The idea behind the if statements, is I need to
        keep the knowledge of who fired for the previous step. To do this I determined which neurons are firing
        using the nonzero function on aas. b and d are the neuron indicies that alternate between being the most
        recent to fire and being one time step behind.
        
        The second set of if statements were made because I needed both b and d to be defined before I started using
        them. In these if statements I iterate through all the neuron indicies that had fired and, depending on
        whether this was the current time step or the previous time step, add or subtract a constant. The opposite
        constant will be added to the the other direction of synaptic connection. For example, if neuron 1 fired then
        neuron 2 the index STDP[1][2] will be strengthened but STDP[2][1] will be weakened.
        
        This model does not however use the exact dependance of time, just the time steps of the iteration. I figured
        this would be acceptable because our activity is currently calculated by these time steps. If the time-
        dependance needs to be exact then more might need to be changed.
        
        The matrix STDP will eventually be used to alter, probably by addition, W. STDP has dimensions of W.
        """        
        
        if tt % 2 == 0:
            a, neuron_number1 = np.nonzero(aas)        
            
        if tt % 2 == 1:
            c, neuron_number2 = np.nonzero(aas)
            
        if tt % 2 == 0 and tt != 0 and neuron_number1 != [] and neuron_number2 !=[]:
            
            for i in neuron_number1:
                for j in neuron_number2:                        
                    STDP[i][j] += stdp_change
                    STDP[j][i] -= stdp_change
        if tt % 2 == 1 and neuron_number1 !=[] and neuron_number2 != []:
          
            for i in neuron_number1:
                for j in neuron_number2:                        
                    STDP[i][j] -= stdp_change 
                    STDP[j][i] += stdp_change
        
        Y += aas
        #update total activity
        Ys[Ys > T] = 0.
        #after firing set back to zero for activity calculations in next time step
        
    return [Y,STDP]

rng = np.random.RandomState(0)

# Parameters
batch_size = 100
num_trials = 25

# Load Images
with open('images.pkl','rb') as f:
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

# Zero timing variables
data_time = 0.
algo_time = 0.

# Initialize the STDP matrix
stdp= np.zeros((M,M))

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
    Y, stdp = activities(X,Q,W,theta)
    muy = np.mean(Y,axis=1)
    Cyy = Y.T.dot(Y)/batch_size
    """
    The following code is the learning rules
    """    
    
    # Update lateral weigts
    dW = alpha*(Cyy-p**2)
    W += dW
    W = W-np.diag(np.diag(W))
    W[W < 0] = 0.

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

    Y_ave = (1.-eta_ave)*Y_ave + eta_ave*muy
    Cyy_ave=(1.-eta_ave)*Cyy_ave + eta_ave*Cyy
    if tt%24 == 0:
        print 'Batch: '+str(tt)+' out of '+str(num_trials)
        print 'Cumulative time spent gathering data: '+str(data_time)+' min'
        print 'Cumulative time spent in SAILnet: '+str(algo_time)+' min'
        print ''
    total_time = data_time+algo_time
print 'Percent time spent gathering data: '+str(data_time/total_time)+' %'
print 'Percent time spent in SAILnet: '+str(algo_time/total_time)+' %'
print ''    

with open('output.pkl','wb') as f:
    cPickle.dump((W,Q,theta),f)


