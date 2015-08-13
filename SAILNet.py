import cPickle
import sys, time, argparse
from Plotter import Plot
from Learning_Rule import Learning_Rule_gpu as Learning_Rule
from Parameters import Parameters
from Network import Network_gpu as Network
from Activity import Activity_gpu as Activity
from Data import Data
from Monitor import Monitor
from Save import make_folder

parameters = Parameters('parameters.txt')  

if sys.argv[1:]:
    parser = argparse.ArgumentParser(description='Process Parameters')
    parser.add_argument('dW_rule',choices = ['dW_SAILnet','dW_identity','dW_time_dep'])
    parser.add_argument('function',choices = ['None','Unit','Step','Well','Gaussian','STDP'])
    args = parser.parse_args()
    parameters.rule = args.dW_rule
    parameters.function = args.function

network = Network(parameters)
activity = Activity(network)
learn = Learning_Rule(network,parameters.rule)
monitor = Monitor(network,learn)
data = Data('/home/jesse/Development/data/vanhateren/whitened_images.h5',
            35,
            parameters.batch_size,
            parameters.N)
#total_time = 0
directory = make_folder(parameters,"Running all rules with all plots")
plotter = Plot(directory)
 
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
    
    
        
    
