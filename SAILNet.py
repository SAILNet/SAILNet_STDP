import cPickle
import os, shutil, time
from Plotter import Plot
from Learning_Rule import Learning_Rule_gpu as Learning_Rule
from Parameters import Parameters
from Network import Network_gpu as Network
from Activity import Activity_gpu as Activity
from Data import Data
from Monitor import Monitor

parameters = Parameters('parameters.txt')  
network = Network(parameters)
activity = Activity(network)
learn = Learning_Rule(network,parameters.rule)
monitor = Monitor(network,learn)
data = Data('/home/jesse/Development/data/vanhateren/whitened_images.h5',
            35,
            parameters.batch_size,
            parameters.N)
total_time = 0

saveAttempt = 0
while os.path.exists('./Trials/' +str(parameters.rule)+ '_Func' + str(parameters.function) + '_Num'+ str(parameters.num_trials) + '_' + str(saveAttempt)):
    saveAttempt += 1
directory = './Trials/' +str(parameters.rule)+ '_Func' + str(parameters.function) + '_Num'+ str(parameters.num_trials) + '_' + str(saveAttempt)
os.makedirs(directory) 
file(directory + '/Comments.txt','wt').write("Testing different plots of Q,Y and X to understand reconstruction behavior. Made all averages over whole matrices.")
shutil.copy2("parameters.txt",directory)

plotter = Plot(directory,parameters)
 
for tt in range(network.parameters.num_trials):
    data.make_X(network) 
    activity.get_acts()
    #start = time.time()
    learn.Update()
    #total_time += time.time()- start
    monitor.log(tt)
    learn.ReduceLearning(tt)
    
    if tt%50 == 0 and tt != 0:
        print('Batch: '+str(tt)+' out of '+ str(parameters.num_trials))
    
    if tt%500 == 0 and tt != 0:
        plotter.Plot_RF(network_Q = network.Q,filenum = tt)
        
#print('Time:' + str(total_time))

network.to_cpu()
monitor.cleanup()
with open(directory +'/data.pkl','wb') as f:
    cPickle.dump((network,learn,monitor),f)

plotter.load_network()
plotter.PlotAll()
    
    
        
    
