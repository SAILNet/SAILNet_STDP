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

def make_folder(params):
    saveAttempt = 0
    base_dir = os.path.join(os.environ['OUTPUT_PATH'],
                            'Trials',
                            str(params.dW_rule)+ '_Func' + str(params.function) + '_Num'+ str(params.num_trials) + '_')
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
    with open(os.path.join(name, 'comments.txt'),'wt') as f:
        f.write(comments)
    return prev,name

def dump_parameters(path, parameters):
    with open(os.path.join(path, 'sailnet_parameters.pkl'),'wb') as f:
        cPickle.dump(parameters, f)
    with open(os.path.join(path, 'sailnet_parameters.pkl'),'rb') as f:
        parameters = cPickle.load(f)
    with open(os.path.join(path, 'sailnet_parameters.txt'),'wt') as f:
        f.write(str(parameters.__dict__))
 
def make_pkl(directory,network,monitor,data_rng):
    temp_file = os.path.join(directory, 'data_temp.pkl')
    final_file = os.path.join(directory, 'data.pkl')
    with open(temp_file,'wb') as f:
        cPickle.dump((network,monitor,data_rng),f)
    shutil.move(temp_file, final_file)

def get_args():
    # Get command-line parameters
    parser = argparse.ArgumentParser(description='SAILNet Parameters')
    parser.add_argument('-f','--folder')                

    parser.add_argument('dW_rule',
                        nargs='?',
                        choices=['dW_SAILnet','dW_identity','dW_time_dep'],
                        default=None)
    parser.add_argument('function',
                        nargs ='?',
                        choices=['None','Unit','Step','Well','Gaussian','STDP','Double_Gaussian','Linear15'],
                        default=None)

    parser.add_argument('-b', '--batch_size', default=None, type=int)
    parser.add_argument('-i', '--num_images', default=None, type=int)
    parser.add_argument('-n', '--num_trials', default=None, type=int)
    parser.add_argument('-t',
                        '--num_iterations',
                        default=None,
                        type=int)
    parser.add_argument('-r',
                        '--reduced_learning_rate',
                        default=None,
                        type=float)
    parser.add_argument('-m', '--norm_infer', default=None, type=bool)

    parser.add_argument('--neurons', default=None,type=int)
    parser.add_argument('--OC', default=None,type=int)
    parser.add_argument('-p', default=None, type=float)

    parser.add_argument('-c', '--comments', default='None')

    return parser.parse_args()

def get_file_params(params_file='sailnet_parameters.txt'):
    return Parameters('sailnet_parameters.txt')          

def final_parameters(file_params, cmd_line_args=None, network_params=None):
    if network_params is not None:
        params = network_params
    else:
        params = file_params
    params.OC = cmd_line_args.OC or params.OC
    params.M = params.N*params.OC
    params.num_trials = cmd_line_args.num_trials or params.num_trials
    params.dW_rule = cmd_line_args.dW_rule or params.dW_rule
    params.function = cmd_line_args.function or params.function
    return params

def load_model():
    args = get_args()
    file_params = get_file_params()
    kwargs = {}
    if args.folder != None:
        assert os.path.exists(args.folder)
        prev, directory = make_subfolder(args.folder,args.comments)
        with open(os.path.join(prev,'data.pkl'),'rb') as f:
            network, _, data_rng = cPickle.load(f)
        kwargs['seed_or_rng'] = data_rng
        network.to_gpu()
        network.current_trial = 0
        parameters = final_parameters(file_params,
                                      cmd_line_args = args,
                                      network_params = network.parameters)
        for attr in ['dW_rule', 'function', 'norm_infer', 'OC', 'N', 'p']:
            if getattr(network.parameters, attr) != getattr(parameters, attr):
                raise ValueError('Value of '+attr+' has changed.')
        network.parameters = parameters
    else:
        parameters = final_parameters(file_params,
                                      cmd_line_args=args)
        network = Network(parameters)
        directory = make_folder(parameters)
        prev, directory = make_subfolder(directory,args.comments)

    dump_parameters(directory, network.parameters)
    
    learn = Learning_Rule(network,parameters.dW_rule)
    monitor = Monitor(network)
    activity = Activity(network)
    data = Data(os.path.join(os.environ['DATA_PATH'],'vanhateren/whitened_images.h5'),
            parameters.num_images,
            parameters.batch_size,
            parameters.N,
            **kwargs)
    plotter = Plot(directory)
    
    return activity,data,directory,learn,monitor,network,parameters,plotter
