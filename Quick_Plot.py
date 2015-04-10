# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 20:20:54 2015

@author: Bernal

This program lets us create new plots based on old Pickle files from previous runs
"""


from SAILnet_Plotting import Plot

OverC = 8
SA = 3

directory = './Trials/OC' + str(OverC) + '_' + str(SA)
data_filename = directory + '/data.pkl'

    
plotter = Plot(data_filename, directory)

#check = plotter.RasterPlot()
