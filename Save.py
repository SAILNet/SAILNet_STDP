# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:31:40 2015

@author: bernal
"""
import os,shutil

def make_folder(parameters,comments):
    saveAttempt = 0
    while os.path.exists('./Trials/' +str(parameters.rule)+ '_Func' + str(parameters.function) + '_Num'+ str(parameters.num_trials) + '_' + str(saveAttempt)):
        saveAttempt += 1
    directory = './Trials/' +str(parameters.rule)+ '_Func' + str(parameters.function) + '_Num'+ str(parameters.num_trials) + '_' + str(saveAttempt)
    os.makedirs(directory) 
    file(directory + '/Comments.txt','wt').write(comments)
    shutil.copy2("parameters.txt",directory)
    
    return directory