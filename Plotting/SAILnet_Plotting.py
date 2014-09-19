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
        with open(fileName,'rb') as f:
            self.W,self.Q,self.theta,self.stdp,self.mag_stdp,self.mag_dW = cPickle.load(f)
    
    
    
    def Plot_RF(self):
        im_size, num_dict = self.Q.shape

        side = int(np.round(np.sqrt(im_size)))
        OC = num_dict/im_size


        self.img = tile_raster_images(Q.T, img_shape = (side,side), tile_shape = (2*side,side*OC/2), tile_spacing=(1, 1), scale_rows_to_unit_interval=True, output_pixel_vals=True)
        plt.imshow(img,cmap=plt.cm.Greys)
        plt.title('Receptive Fields')
        plt.imsave('RF_no_stdp.png', img, cmap=plt.cm.Greys)
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









# <codecell>

just_dW=Run("output_no_stdp.pkl")
just_stdp=Run("just_stdp.pkl")
stdp_and_dW=Run("stdp_and_dW.pkl")

#just_dW.Plot_RF()

plt.figure(just_dW.PlotdWstdp("Magnitudes of dW and STDP Matricies Using dW"))


plt.figure(just_stdp.PlotdWstdp("Magnitudes of dW and STDP Matricies Using STDP"))

plt.figure(just_stdp.Plotstdp("Using Just STDP"))

plt.figure(just_stdp.PlotdW("Using Just STDP"))



plt.figure(stdp_and_dW.PlotdWstdp("Using Both STDP and dW"))



plt.figure(stdp_and_dW.Plotstdp("Using Both dW and STDP"))

plt.figure(stdp_and_dW.PlotdW("Using Both dW and STDP"))
"""


    






Plot_RF(Q)

plt.close
plt.clf


PlotdWstdp(magnitude_stdp,magnitude_dW)





with open('output_stdp.pkl','rb') as f:
    W_stdp,Q,theta,stdp,magnitude_stdp,magnitude_dW = cPickle.load(f)

"""


