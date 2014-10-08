# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# SAILnet Testing

# <codecell>

import cPickle
import matplotlib.pyplot as plt
import numpy as np
from utils import tile_raster_images
#%matplotlib inline 

# <codecell>

class Run():
    
    def __init__(self,fileName):
        self.fileName=fileName
        with open(self.fileName,'rb') as f:
            self.W,self.Q,self.theta,self.stdp,self.mag_stdp,self.mag_dW, self.correlation, self.Yavg,self.Cavg, self.time_dep = cPickle.load(f)
    
    
    
    def Plot_RF(self):
        im_size, num_dict = self.Q.shape

        side = int(np.round(np.sqrt(im_size)))
        OC = num_dict/im_size


        img = tile_raster_images(self.Q.T, img_shape = (side,side), tile_shape = (2*side,side*OC/2), tile_spacing=(1, 1), scale_rows_to_unit_interval=True, output_pixel_vals=True)
        plt.imshow(img,cmap=plt.cm.Greys)
        plt.title('Receptive Fields - '+self.fileName[:len(self.fileName)-4])
        plt.imsave('Images/RF '+self.fileName[:len(self.fileName)-4]+'.png', img, cmap=plt.cm.Greys)
        plt.show()
        plt.close
        plt.clf


    def PlotdWstdp(self):
        
        plt.plot(self.mag_stdp, color="green", label="STDP")
        plt.plot(self.mag_dW,color="blue", label="dW")
        plt.legend(bbox_to_anchor=(1,.5))
        plt.title("Magnitude of STDP and dW "+self.fileName[:len(self.fileName)-4])
        plt.xlabel("Number of Trials")
        plt.savefig('Images/stdp_dW_'+self.fileName[:len(self.fileName)-4]+'.png')
        plt.clf
        
    def Plotstdp(self):
        plt.plot(self.mag_stdp, color="green", label="STDP")
        plt.title("Magnitude of STDP"+self.fileName[:len(self.fileName)-4])
        plt.xlabel("Number of Trials")
        plt.savefig('Images/Magnitude STDP Using'+self.fileName[:len(self.fileName)-4]+'.png')
    
    def PlotdW(self):
        plt.plot(self.mag_dW,color="blue", label="dW")
        plt.title("Magnitude of dW"+self.fileName[:len(self.fileName)-4])
        plt.xlabel("Number of Trials")
        plt.savefig('Images/Magnitude dW Using'+self.fileName[:len(self.fileName)-4]+'.png')
        
    def PlotYavg(self):
        
        plt.plot(self.Yavg, color="brown")
        plt.title(self.fileName[:len(self.fileName)-4] + 'Y_avg')
        plt.xlabel("Number of Trials")
        plt.savefig('Images/Y_avg'+self.fileName[:len(self.fileName)-4]+'.png')
    
    def PlotCavg(self):
        
        plt.plot(self.Cavg, color="red")
        plt.title(self.fileName[:len(self.fileName)-4] + 'C_avg')
        plt.xlabel("Number of Trials")
        plt.savefig('Images/C_avg'+ self.fileName[:len(self.fileName)-4] + '.png')
        
    def Plotcor(self):
        
        plt.plot(self.correlation, color = "purple")
        plt.title(self.fileName[:len(self.fileName)-4] + 'Correlation of dW and STDP')
        plt.xlabel("Number of Trials")
        plt.savefig('Images/Correlation of dW and STDP'+self.fileName[:len(self.fileName)-4] +'.png')

    def PlotTimeDep(self):
        plt.plot(self.time_dep[25])
        plt.title(self.fileName[:len(self.fileName)-4] + 'Time Weighting Matrix')
        
    
    def PlotAll(self):
        plt.figure(self.Plot_RF())
        plt.figure(self.PlotdWstdp())
        plt.figure(self.PlotdW())
        plt.figure(self.Plotstdp())
        plt.figure(self.Plotcor())
        plt.figure(self.PlotCavg())
        plt.figure(self.PlotYavg())
        plt.figure(self.PlotTimeDep())






# <codecell>

#stdp5000New=Run("stdp5000model_New.pkl")
#stdp25000New=Run("stdp25000model_New.pkl")
dW25000New=Run("dW25000model_New.pkl")


#stdp25000New.PlotAll()
dW25000New.PlotAll()


