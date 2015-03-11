import numpy as np
import cPickle, time
from pca import pca
import van_hateren as VH
from utils import tile_raster_images
import matplotlib.pyplot as plt
import os
import shutil
from SAILnet_Plotting import Plot
from Network import Network
from Activity import Activity
from Exp_STDP_rule import Exp_STDP
from SAILNet_rule import SAILNet_rule

    

    
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
learn = SAILNet_rule()

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



# Zero timing variables
data_time = 0.
algo_time = 0.






#Correlation matrix for each neuron

#cor_dW_stdp=np.zeros_like(mag_dW)



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
    
    
    """
    using stdp matrix to update W
    """
    
    time_stdp=time.time()
    
    learn.CalculateChange(network)
    
    time_stdp= time.time()-time_stdp
    
    """
    The following code is the learning rules
    """    
    
    # Update lateral weigts
    #
    
    learn.Update(network)
    

    
    dt = time.time()-dt
    algo_time += dt/60.
    
    
    """
    Updating all the variables which store important information for analysis
    """
    
    network.UpdateData(tt,learn)
    
    
    """
    Reducing step size after 5000 trials
    """
    network.ReduceLearning(tt)
    
    """
    Saving Images for RF gif
    """
    if create_gif and tt%trials_per_image==0:
        gif(network.Q,tt)
    
    if tt%50 == 0 and tt != 0:
        print 'Batch: '+str(tt)+' out of '+str(network.num_trials)
        print 'Cumulative time spent gathering data: '+str(data_time)+' min'
        print 'Cumulative time spent in SAILnet: '+str(algo_time)+' min'
        #print 'Cumulative time spent calculating STDP weights: '+str(time_for_stdp1)+' min'
        print ''
    total_time = data_time+algo_time
    
    
print 'Percent time spent gathering data: '+str(data_time/total_time*100)+' %'
print 'Percent time spent in SAILnet: '+str(algo_time/total_time*100)+' %'
#print 'Percent time spent calculating STDP: '+str(time_for_stdp1/total_time*100)+' %'
print '' 

saveAttempt = 0   
    
while os.path.exists("./Trials/OC"+str(network.OC)+'_'+str(saveAttempt)):
    saveAttempt += 1
    
directory = "./Trials/OC"+str(network.OC)+'_'+str(saveAttempt)
os.makedirs(directory) 
    
shutil.copy2("parameters.txt",directory)
with open(directory +'/data.pkl','wb') as f:
    cPickle.dump((network,learn),f)

data_filename = directory + '/data.pkl'

plotter = Plot(data_filename, directory)

print network.Y

plotter.PlotAll()
    
    
        
    
