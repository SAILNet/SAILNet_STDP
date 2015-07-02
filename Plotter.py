import matplotlib.pyplot as plt
import matplotlib.font_manager as fnt
import numpy as np
from utils import tile_raster_images
import os


class Plotter():
    
    def __init__(self, monitor):
        self.monitor = monitor
        
    def plot_ybar(self):
        plt.plot(self.monitor.y_bar)
        plt.title("Average Y")
        
    def plot_cyybar(self):
        plt.plot(self.monitor.Cyy_bar)
        plt.title("AverageY^2")
        
    def plot_dW(self):
        plt.plot(self.monitor.dW)
        plt.title("Average dW")