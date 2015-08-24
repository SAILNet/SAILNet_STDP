# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:31:40 2015

@author: bernal
"""
import argparse, cPickle, os, shutil
from plotter import Plot
from learning_rule import Learning_Rule
from parameters import Parameters
from network import Network 
from activity import Activity
from data import Data
from monitor import Monitor

def make_folder(parameters):
    saveAttempt = 0
    base_dir = os.path.join(os.environ['OUTPUT_PATH'],
                            'Trials',
                            str(parameters.rule)+ '_Func' + str(parameters.function) + '_Num'+ str(parameters.num_trials) + '_')
    while os.path.exists(base_dir + str(saveAttempt)):
        saveAttempt += 1
    directory = base_dir + str(saveAttempt)
    os.makedirs(directory) 
    
    return directory    
    
def make_subfolder(directory,comments):
    dirnames = []
    for d,ds,fs in os.walk(directory):
        dirnames = sorted(ds)
        break
    if len(dirnames) == 0:
        prev = None        
        name = os.path.join(directory,'000000')        
        os.makedirs(name)
    else:
        prev = os.path.join(directory,str(int(dirnames[-1])).zfill(6))
        name = os.path.join(directory,str(int(dirnames[-1])+1).zfill(6))
        os.makedirs(name)
    with open(name + '/Comments.txt','wt') as f:
        f.write(comments)
    shutil.copy2("sailnet_parameters.txt",name)
    
    return prev,name


def make_pkl(directory,network,monitor,data_rng):
    temp_file = os.path.join(directory, 'data_temp.pkl')
    final_file = os.path.join(directory, 'data.pkl')
    with open(temp_file,'wb') as f:
        cPickle.dump((network,monitor,data_rng),f)
    shutil.move(temp_file, final_file)
    
def load_model():    
    parameters = Parameters('sailnet_parameters.txt')          

    parser = argparse.ArgumentParser(description='Process Parameters')
    parser.add_argument('-f','--folder')                
    parser.add_argument('dW_rule',nargs='?',choices = ['dW_SAILnet','dW_identity','dW_time_dep'],default = parameters.rule)
    parser.add_argument('function',nargs ='?',choices = ['None','Unit','Step','Well','Gaussian','STDP','Negative','Linear','Double_Gaussian'],default= parameters.function)
    parser.add_argument('-n','--num_trials',default=parameters.num_trials,type=int)
    parser.add_argument('-p',default=parameters.p,type=float)
    parser.add_argument('-b','--batch_size',default=parameters.batch_size,type=int)
    parser.add_argument('-r','--reduced_learning_rate',default=parameters.reduced_learning_rate,type=float)
    parser.add_argument('--OC',default=parameters.OC,type=int)
    parser.add_argument('-c','--comments',default='None')
    args = parser.parse_args()
    parameters.OC = args.OC
    parameters.M = parameters.N*parameters.OC
    parameters.num_trials = args.num_trials
    parameters.rule = args.dW_rule
    parameters.function = args.function
    parameters.keep_spikes()
    kwargs = {}
    if args.folder != None:
        os.path.exists(args.folder)
        prev,directory = make_subfolder(args.folder,args.comments)
        with open(os.path.join(prev,'data.pkl'),'rb') as f:
            network,_,data_rng = cPickle.load(f)
        kwargs['seed_or_rng'] = data_rng
        network.to_gpu()
        network.current_trial = 0
        for attr in ['rule', 'function']:
            if getattr(network.parameters, attr) != getattr(parameters, attr):
                raise ValueError('Value of '+attr+' has changed.')
        network.parameters = parameters
    else:
        network = Network(parameters)
        directory = make_folder(parameters)
        prev, directory = make_subfolder(directory,args.comments)
    print(network.parameters.rule)
    print(network.parameters.function)
    
    learn = Learning_Rule(network,parameters.rule)    
    monitor = Monitor(network)
    activity = Activity(network)
    data = Data(os.path.join(os.environ['DATA_PATH'],'vanhateren/whitened_images.h5'),
            parameters.num_images,
            parameters.batch_size,
            parameters.N,
            **kwargs)
    plotter = Plot(directory)
    
    return activity,data,directory,learn,monitor,network,parameters,plotter
