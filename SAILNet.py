import Save

activity,data,directory,learn,monitor,network,parameters,plotter = Save.load_model()

while network.continue_learning():
    tt=network.current_trial
    data.make_X(network) 
    activity.get_acts()
    #start = time.time()
    monitor.log(tt)
    learn.Update()
    #total_time += time.time()- start
    learn.ReduceLearning(tt)
    
    if tt%50 == 0 and tt != 0:
        print('Batch: '+str(tt)+' out of '+ str(parameters.num_trials))
    
    if tt%500 == 0 and tt != 0:
        plotter.Plot_RF(network_Q = network.Q,filenum = tt)
                
#print('Time:' + str(total_time))

network.to_cpu()
monitor.cleanup()
Save.make_pkl(directory,network,monitor,data.rng)

plotter.load_network()
plotter.PlotAll()
    
    
        
    
