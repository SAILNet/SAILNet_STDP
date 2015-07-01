import numpy as np
import cPickle
import os, shutil
from SAILnet_Plotting import Plot
from Plotter import Plotter
import Learning_Rule
from Parameters import Parameters
from Network import Network_gpu as Network
from Activity import Activity_gpu as Activity
from Utility import Data
from Monitor import Monitor

parameters = Parameters('parameters.txt')
network = Network(parameters)
activity = Activity(network)
learn = getattr(Learning_Rule,parameters.rule)(network)
monitor = Monitor(network)
data = Data('/home/jesse/Development/data/vanhateren/whitened_images.h5',
            35,
            parameters.batch_size,
            parameters.N)

"""Bolean, Save RF fields and create gif
create_gif=False
trials_per_image=10
gif_images=np.zeros(network.num_trials/trials_per_image)
"""

for tt in range(network.parameters.num_trials):
    data.make_X(network) 
    activity.get_acts()
    learn.Update()
    monitor.log()
    learn.ReduceLearning(tt)
    
    """
    Saving Images for RF gif
    """
    #if create_gif and tt%trials_per_image==0:
    #    gif(network.Q,tt)
    
    if tt%50 == 0 and tt != 0:
        print('Batch: '+str(tt)+' out of '+ str(network.parameters.num_trials))

plott = Plotter(monitor)

plott.plot_ybar()

saveAttempt = 0
    
while os.path.exists("./Trials/OC"+str(network.parameters.OC)+'_'+str(saveAttempt)):
    saveAttempt += 1

directory = "./Trials/OC" +str(network.parameters.OC)+'_'+str(saveAttempt)
os.makedirs(directory) 
    
shutil.copy2("parameters.txt",directory)
network.to_cpu()
with open(directory +'/data.pkl','wb') as f:
    cPickle.dump((network,monitor),f)

data_filename = directory + '/data.pkl'

plotter = Plot(data_filename, directory)

plotter.PlotAll()
    
    
        
    
