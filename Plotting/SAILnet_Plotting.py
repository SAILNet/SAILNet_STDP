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
            self.W,self.Q,self.theta,self.stdp,self.mag_stdp,self.mag_dW, self.correlation, self.Yavg,self.Cavg = cPickle.load(f)
    
    
    
    def Plot_RF(self,name):
        im_size, num_dict = self.Q.shape

        side = int(np.round(np.sqrt(im_size)))
        OC = num_dict/im_size


        img = tile_raster_images(self.Q.T, img_shape = (side,side), tile_shape = (2*side,side*OC/2), tile_spacing=(1, 1), scale_rows_to_unit_interval=True, output_pixel_vals=True)
        plt.imshow(img,cmap=plt.cm.Greys)
        plt.title(name + 'Receptive Fields')
        plt.imsave(name+'.png', img, cmap=plt.cm.Greys)
        plt.show()
        plt.close
        plt.clf


    def PlotdWstdp(self,title):
        
        plt.plot(self.mag_stdp, color="green", label="STDP")
        plt.plot(self.mag_dW,color="blue", label="dW")
        plt.legend(bbox_to_anchor=(1,.5))
        plt.title(title)
        plt.xlabel("Number of Trials")
        plt.savefig('stdp_dW.png')
        plt.clf
        
    def Plotstdp(self,title):
        plt.plot(self.mag_stdp, color="green", label="STDP")
        plt.title(title)
        plt.xlabel("Number of Trials")
    
    def PlotdW(self,title):
        plt.plot(self.mag_dW,color="blue", label="dW")
        plt.title(title)
        plt.xlabel("Number of Trials")
        
    def PlotYavg(self):
        
        plt.plot(self.Yavg, color="yellow")
        plt.title(self.fileName + 'Y_avg')
        plt.xlabel("Number of Trials")
    
    def PlotCavg(self):
        
        plt.plot(self.Cavg, color="red")
        plt.title(self.fileName + 'C_avg')
        plt.xlabel("Number of Trials")
        
    def Plotcor(self):
        
        plt.plot(self.correlation, color = "purple")
        plt.title(self.fileName + 'Correlation of dW and STDP')
        plt.xlabel("Number of Trials")








# <codecell>

#just_dW=Run("output_no_stdp.pkl")
stdp1000=Run("stdp1000.pkl")
#stdp_and_dW=Run("stdp_and_dW.pkl")
dW1000=Run("dW1000.pkl")



plt.figure(stdp1000.PlotdWstdp("Magnitudes of dW and STDP Matricies Using STDP"))

plt.figure(stdp1000.Plotstdp("Using Just STDP"))

plt.figure(stdp1000.PlotdW("Using Just STDP"))

#plt.figure(stdp1000.Plot_RF("RF_Just_STDP"))

plt.figure(stdp1000.Plotcor())


    





"""
Plot_RF(Q)

plt.close
plt.clf


PlotdWstdp(magnitude_stdp,magnitude_dW)





with open('output_stdp.pkl','rb') as f:
    W_stdp,Q,theta,stdp,magnitude_stdp,magnitude_dW = cPickle.load(f)

"""


