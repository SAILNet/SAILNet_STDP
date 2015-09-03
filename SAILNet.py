import save

activity,data,directory,learn,monitor,network,parameters,plotter = save.load_model()

while network.continue_learning():
    tt=network.current_trial
    data.make_X(network) 
    activity.get_acts()
    #start = time.time()
    monitor.log(tt)
    learn.Update()
    #total_time += time.time()- start
    learn.ReduceLearning(tt)
    
    if (tt+1)%50 == 0:
        print('Batch: '+str(tt+1)+' out of '+ str(parameters.num_trials))
    
    if (tt+1)%500 == 0:
        plotter.Plot_RF(network_Q = network.Q,filenum = tt)
    if (tt+1)%1000 == 0:
        save.make_pkl(directory,network,monitor,data.rng)
                
#print('Time:' + str(total_time))

network.to_cpu()
monitor.cleanup()
print('Saving...')
save.make_pkl(directory,network,monitor,data.rng)

print('Plotting...')
plotter.load_network()
plotter.PlotAll()
