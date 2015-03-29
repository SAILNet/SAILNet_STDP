# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 20:20:54 2015

@author: Bernal

This program lets us create new plots based on old Pickle files from previous runs
"""

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
from Learning_Rule import Exp_STDP,SAILNet_rule

OverC = 4
SA = 0

directory = '/Trials/OC' + str(OverC) + '_' + str(SA)
data_filename = directory + '/data.pkl'

config_file = 'parameters.txt'

network = Network(config_file)

learn = SAILNet_rule()

with open(directory +'/data.pkl','wb') as f:
    cPickle.dump((network,learn),f)    
    
plotter = Plot(data_filename, directory)


