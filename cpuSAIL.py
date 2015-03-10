import numpy as np
import cPickle, time
from pca import pca
import van_hateren as VH
from utils import tile_raster_images
import matplotlib.pyplot as plt
import ConfigParser
import os
import shutil
from SAILnet_Plotting import Plot
from Network import Network
from Activity import Activity

    
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

config_file = 'parameters.txt'

network = Network(config_file)
activity = Activity()

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


sz = np.sqrt(network.N).astype(np.int)

Y_ave = network.p
Cyy_ave = network.p**2


Cyy_ave_pertrial=np.zeros(network.num_trials)
Y_ave_pertrial=np.zeros_like(Cyy_ave_pertrial)

# Zero timing variables
data_time = 0.
algo_time = 0.



time_for_stdp=time.time()
stdp=np.zeros((network.M,network.M))
stdp_model="New"

time_dep=STDP(network.M,stdp_model,network.batch_size)

time_for_stdp= time.time()-time_for_stdp

#The following will keep track of the change of the magnitude of the stdp
#matrix for each trial.

mag_stdp=np.zeros(network.num_trials)

#mag_dW will track the magnitude changes in dW

mag_dW=np.zeros_like(mag_stdp)

#mag_W will track the magnitude in W

mag_W = np.zeros_like(mag_stdp)

#Correlation matrix for each neuron

cor_dW_stdp=np.zeros_like(mag_stdp)

reconstruction_error=np.zeros_like(mag_dW)

#Bolean, Save RF fields and create gif
create_gif=False
trials_per_image=10
gif_images=np.zeros(network.num_trials/trials_per_image)



for tt in xrange(network.num_trials):
    # Extract image patches from images
    dt = time.time()
    for ii in xrange(network.batch_size):
        r = BUFF+int((imsize-sz-2.*BUFF)*rng.rand())
        c = BUFF+int((imsize-sz-2.*BUFF)*rng.rand())
        myimage = images[int(num_images*rng.rand()),r:r+sz,c:c+sz].ravel()
        #takes a chunck from a random image, size of 16X16 patch at a random location       
        
        
        network.X[ii] = myimage
        #creating a list of image patches to work with
    
    #Conducts Principle Component Analysis
    pca_instance.fit(network.X)
    network.X=pca_instance.transform_zca(network.X)
    #Forces mean to be 0    
    network.X = network.X-np.mean(network.X)
    network.X = network.X/network.X.std()
    
    dt = time.time()-dt
    data_time += dt/60.
    

    dt = time.time()
    # Calcuate network activities
    
    activity.get_acts(network)
    
    """
    This commented out section was used to determine the sign for time_dep
    activity_log=np.zeros((batch_size,M,iterations))
    activity_log[0][0][0]+=1
    activity_log[0][10][1]+=1
    """
    
    muy = np.mean(network.Y,axis=0)
    Cyy = network.Y.T.dot(network.Y)/network.batch_size
    
    """
    using stdp matrix to update W
    """
    
    time_stdp=time.time()
    
    for batch in xrange(network.batch_size):
        stdp+=np.dot(network.spike_train[batch],np.dot(time_dep,network.spike_train[batch].T))
    stdp = stdp/network.batch_size
    time_stdp= time.time()-time_stdp
    
    time_for_stdp+= time_stdp
    
    mag_stdp[tt]=np.linalg.norm(stdp)
    
    """
    The following code is the learning rules
    """    
    
    # Update lateral weigts
    dW = network.alpha*(Cyy-network.p**2)
    network.W += stdp
    network.W -= network.lateral_constraint*network.W
    network.W = network.W-np.diag(np.diag(network.W))
    network.W[network.W < 0] = 0.
    
    mag_dW[tt]=np.linalg.norm(dW)
    mag_W[tt] =np.linalg.norm(network.W)

    # Update feedforward weights
    square_act = np.sum(network.Y*network.Y,axis=0)
    mymat = np.diag(square_act)
    dQ = network.beta*network.X.T.dot(network.Y)/network.batch_size - network.beta*network.Q.dot(mymat)/network.batch_size
    network.Q += dQ

    # Update thresholds
    dtheta = network.gamma*(np.sum(network.Y,axis=0)/network.batch_size-network.p)
    network.theta += dtheta
    dt = time.time()-dt
    algo_time += dt/60.
    time_for_stdp1= time_for_stdp/60
    
    
    """
    We shall determine the correlation between dW and stdp by dW*stdp/(|dW||stdp|)
    """
    cor_dW_stdp[tt]=sum(sum(dW.dot(stdp)))/(np.linalg.norm(dW)*np.linalg.norm(stdp))
    
    #Error in reconstucting the images
    reconstruction_error[tt]=np.sum(np.sum((network.X-network.Y.dot(network.Q.T))**2))/(2*network.N*network.batch_size)  
    
    Y_ave = (1.-network.eta_ave)*Y_ave + network.eta_ave*muy
    Cyy_ave=(1.-network.eta_ave)*Cyy_ave + network.eta_ave*Cyy
    Cyy_ave_pertrial[tt]=sum(sum(Cyy-np.diag(np.diag(Cyy))))/(network.N**2-network.N)
    Y_ave_pertrial[tt]=np.mean(Y_ave)
    
    """
    Reducing step size after 5000 trials
    """
    if tt >= 5000:
        network.gamma=network.gamma*network.reduced_learning_rate
        network.alpha=network.alpha*network.reduced_learning_rate
        network.beta=network.beta*network.reduced_learning_rate
    """
    Saving Images for RF gif
    """
    if create_gif and tt%trials_per_image==0:
        gif(network.Q,tt)
    
    if tt%50 == 0 and tt != 0:
        print 'Batch: '+str(tt)+' out of '+str(network.num_trials)
        print 'Cumulative time spent gathering data: '+str(data_time)+' min'
        print 'Cumulative time spent in SAILnet: '+str(algo_time)+' min'
        print 'Cumulative time spent calculating STDP weights: '+str(time_for_stdp1)+' min'
        print ''
    total_time = data_time+algo_time+time_for_stdp1
    
    
print 'Percent time spent gathering data: '+str(data_time/total_time*100)+' %'
print 'Percent time spent in SAILnet: '+str(algo_time/total_time*100)+' %'
print 'Percent time spent calculating STDP: '+str(time_for_stdp1/total_time*100)+' %'
print '' 

saveAttempt = 0   
    
while os.path.exists("./Trials/OC"+str(network.OC)+'_'+str(saveAttempt)):
    saveAttempt += 1
    
directory = "./Trials/OC"+str(network.OC)+'_'+str(saveAttempt)
os.makedirs(directory) 
    
shutil.copy2("parameters.txt",directory)
with open(directory +'/data.pkl','wb') as f:
    cPickle.dump((network.W,network.Q,network.theta,stdp,mag_stdp,mag_dW,cor_dW_stdp,
                  Y_ave_pertrial,Cyy_ave_pertrial,time_dep,
                  reconstruction_error, mag_W),f)

data_filename = directory + '/data.pkl'

plotter = Plot(data_filename, directory)

print network.Y

plotter.PlotAll()
    
    
        
    
