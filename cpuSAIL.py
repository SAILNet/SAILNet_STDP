import numpy as np
import cPickle, time
from math import ceil
from pca import pca
import van_hateren as VH
from utils import tile_raster_images
import matplotlib.pyplot as plt


def activities(X,Q,W,theta):
    batch_size, N = X.shape
    sz = int(np.sqrt(N))

    M = Q.shape[1]
    
    """
    Q is the matrix of connection strengths from each input to each neuron. it is an (Inputs X number of neurons) matrix
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
        
        
        
        #Forces mean to be 0
        Y += aas
        #update total activity
        Ys[Ys > T] = 0.
        #after firing set back to zero for activity calculations in next time step
    
        
    return [Y,stdp]
    
def STDP(M,model,iterations):
    
    
    time_dep= np.zeros((iterations,iterations))

    if model == "New":
       post_activity=-.027
       pre_activity=.027
       time_scale=4
       for i in xrange(iterations):
            for j in xrange(iterations):
                
                dt=i-j
                #i-j gives the correct signs to strengthen pre to post synaptic activity 10/05/14
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
                    dt=i-j
                    #i-j gives the correct signs to strengthen pre to post synaptic activity 10/05/14
                    if np.sign(dt) == 1:
                        time_dep[i][j]+= pre_activity*np.exp(-abs(dt/time_scale))
                    else:
                        time_dep[i][j]+= post_activity*np.exp(-abs(dt/time_scale))
                else:
                    time_dep[i][j]=0
    
 
                
    return time_dep
    
def gif(Q,iteration):
    im_size, num_dict = Q.shape

    side = int(np.round(np.sqrt(im_size)))
    OC = num_dict/im_size


    img = tile_raster_images(Q.T, img_shape = (side,side), tile_shape = (2*side,side*OC/2), tile_spacing=(1, 1), scale_rows_to_unit_interval=True, output_pixel_vals=True)
    plt.imsave('Plotting/Images/gif/RF '+ str(iteration)+ '.png', img, cmap=plt.cm.Greys)
    

rng = np.random.RandomState(0)

# Parameters
batch_size = 50
num_trials = 25000

reduced_learning_rate=.99985

#Load Images in the Van Hateren Image set.
van_hateren_instance=VH.VanHateren("vanhateren_iml\\")
images=van_hateren_instance.load_images(100)
num_images, imsize, imsize = images.shape

#Creat PCA Instance
pca_instance=pca.PCA(whiten=True)

"""
# Load Images, for smaller image set
with open('images.pkl','r') as f:
    images = cPickle.load(f)
imsize, imsize, num_images = images.shape
images = np.transpose(images,axes=(2,0,1))
"""
BUFF = 20

# Neuron Parameters
N = 256
sz = np.sqrt(N).astype(np.int)
OC = 8 #Over-Completeness: num of neurons = OC * num of inputs
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
Y_ave_pertrial=np.zeros_like(Cyy_ave_pertrial)

# Zero timing variables
data_time = 0.
algo_time = 0.



time_for_stdp=time.time()
stdp=np.zeros((M,M))
stdp_model="New"

time_dep=STDP(M,stdp_model,batch_size)

time_for_stdp= time.time()-time_for_stdp

#The following will keep track of the change of the magnitude of the stdp
#matrix for each trial.

mag_stdp=np.zeros(num_trials)

#mag_dW will track the magnitude changes in dW

mag_dW=np.zeros_like(mag_stdp)

#Correlation matrix for each neuron

cor_dW_stdp=np.zeros_like(mag_stdp)

reconstruction_error=np.zeros_like(mag_dW)

#Bolean, Save RF fields and create gif
create_gif=False
trials_per_image=10
gif_images=np.zeros(num_trials/trials_per_image)

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
        
        
        X[ii] = myimage
        #creating a list of image patches to work with
    
    #Conducts Principle Component Analysis
    pca_instance.fit(X)
    X=pca_instance.transform_zca(X)
    #Forces mean to be 0    
    X = X-np.mean(X)
    X = X/X.std()
    
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
    
    #mag_stdp[tt]=np.linalg.norm(stdp)
    
    """
    The following code is the learning rules
    """    
    
    # Update lateral weigts
    dW = alpha*(Cyy-p**2)
    W += stdp
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
    time_for_stdp1= time_for_stdp/60
    
    """
    We shall determine the correlation between dW and stdp by dW*stdp/(|dW||stdp|)
    """
    cor_dW_stdp[tt]=sum(sum(dW.dot(stdp)))/(np.linalg.norm(dW)*np.linalg.norm(stdp))
    
    #Error in reconstucting the images
    reconstruction_error[tt]=np.sum(np.sum((X-Y.dot(Q.T))**2))/(2*N*batch_size)  
    
    Y_ave = (1.-eta_ave)*Y_ave + eta_ave*muy
    Cyy_ave=(1.-eta_ave)*Cyy_ave + eta_ave*Cyy
    Cyy_ave_pertrial[tt]=sum(sum(Cyy-np.diag(np.diag(Cyy))))/(N**2-N)
    Y_ave_pertrial[tt]=np.mean(Y_ave)
    
    """
    Reducing step size after 5000 trials
    """
    if tt >= 5000:
        gamma=gamma*reduced_learning_rate
        alpha=alpha*reduced_learning_rate
        beta=beta*reduced_learning_rate
    """
    Saving Images for RF gif
    """
    if create_gif and tt%trials_per_image==0:
        gif(Q,tt)
    
    if tt%50 == 0 and tt != 0:
        print 'Batch: '+str(tt)+' out of '+str(num_trials)
        print 'Cumulative time spent gathering data: '+str(data_time)+' min'
        print 'Cumulative time spent in SAILnet: '+str(algo_time)+' min'
        print 'Cumulative time spent calculating STDP weights: '+str(time_for_stdp1)+' min'
        print ''
    total_time = data_time+algo_time+time_for_stdp1
print 'Percent time spent gathering data: '+str(data_time/total_time*100)+' %'
print 'Percent time spent in SAILnet: '+str(algo_time/total_time*100)+' %'
print 'Percent time spent calculating STDP: '+str(time_for_stdp1/total_time*100)+' %'
print '' 

with open('Plotting/NewSTDP' + str(num_trials)+'OC_'+str(OC)+'.pkl','wb') as f:
    cPickle.dump((W,Q,theta,stdp,mag_stdp,mag_dW,cor_dW_stdp,Y_ave_pertrial,Cyy_ave_pertrial,time_dep,reconstruction_error),f)
    
    
        
    
