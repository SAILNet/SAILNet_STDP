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
    
    if (tt)%50 == 0:
        print('Batch: '+str(tt)+' out of '+ str(parameters.num_trials))
    
    if (tt)%500 == 0:
        plotter.plot_rf(network_Q = network.Q,filenum = tt)
    
    if parameters.plot_interval != None:
        if (tt)%parameters.plot_interval == 0 and tt != 0:
            
            print('Saving...')
            save.make_pkl(directory, network, monitor, data.rng)

            print('Plotting...')
            plotter.load_network(network, monitor)
            plotter.PlotAll()

#print('Time:' + str(total_time))

print('Saving...')
save.make_pkl(directory, network, monitor, data.rng)

print('Plotting...')
plotter.load_network()
plotter.PlotAll()
