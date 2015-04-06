# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# SAILnet Testing

# <codecell>

import cPickle
import matplotlib.pyplot as plt
import numpy as np
from utils import tile_raster_images
import os
#%matplotlib inline 

# <codecell>

class Plot():
    
    def __init__(self,fileName, directory):
        self.fileName=fileName
        self.directory = directory
        if os.path.exists(self.directory+'/Images')==False:       
            os.makedirs(self.directory+'/Images')
        with open(self.fileName,'rb') as f:
            self.network, self.learning_rule = cPickle.load(f)
            
    def Plot_RF(self):
        im_size, num_dict = self.network.Q.shape

        side = int(np.round(np.sqrt(im_size)))
        OC = num_dict/im_size


        img = tile_raster_images(self.network.Q.T, img_shape = (side,side),
                                 tile_shape = (2*side,side*OC/2), tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=True, output_pixel_vals=True)
        plt.figure(0)
        plt.imshow(img,cmap=plt.cm.Greys)
        plt.title('Receptive Fields with 25000 Iterations and STDP Learning Rule')
        plt.imsave(self.directory + '/Images/Magnitude_W.png', img, cmap=plt.cm.Greys)
        plt.show()
        plt.close
        plt.clf

    """
    def PlotdWstdp(self):
        plt.figure(1)
        plt.plot(self.mag_stdp, color="green", label="STDP")
        plt.plot(self.mag_dW,color="blue", label="dW")
        plt.legend(bbox_to_anchor=(1,.5))
        plt.title("Magnitude of STDP and dW with 25000 Iterations and STDP Learning Rule")
        plt.xlabel("Number of Trials")
        plt.savefig(self.directory + '/Images/Magnitude_dW_STDP.png')
        plt.clf
    """
    """
    def Plotstdp(self):
        plt.figure(2)
        plt.plot(self.mag_stdp, color="green", label="STDP")
        plt.title("Magnitude of STDP with 25000 Iterations and STDP Learning Rule")
        plt.xlabel("Number of Trials")
        plt.savefig(self.directory + '/Images/Magnitude_STDP.png')
    """
    def PlotdW(self):
        plt.figure(3)
        plt.plot(self.network.mag_dW,color="blue", label="dW")
        plt.title("Magnitude of dW with 25000 Iterations and STDP Learning Rule ")
        plt.xlabel("Number of Trials")
        plt.savefig(self.directory + '/Images/Magnitude_dW.png')
        
    def PlotYavg(self):
        plt.figure(4)
        plt.plot(self.network.Y_ave_pertrial, color="brown")
        plt.title('Y_avg with 25000 Iterations and STDP Learning Rule')
        plt.xlabel("Number of Trials")
        plt.savefig(self.directory + '/Images/Yavg.png')
    
    def PlotCavg(self):
        plt.figure(5)
        plt.plot(self.network.Cyy_ave_pertrial, color="red")
        plt.title('C_avg with 25000 Iterations and STDP Learning Rule')
        plt.xlabel("Number of Trials")
        plt.savefig(self.directory + '/Images/Cavg.png')
    """
    def Plotcor(self):
        plt.figure(6)
        plt.plot(self.correlation, color = "purple")
        plt.title('Correlation of dW and STDP with 25000 Iterations and STDP Learning Rule')
        plt.xlabel("Number of Trials")
        plt.savefig(self.directory + '/Images/Correlation_dW_STDP.png')
    """
    
    def PlotTimeDep(self):
        try:
            plt.figure(7)
            plt.plot(self.learning_rule.time_dep[25])
            plt.title(self.fileName[:len(self.fileName)-4] + 'Time Weighting Matrix')
            plt.savefig(self.directory + '/Images/Weighting_Matrix.png')
        except(AttributeError):
            pass
        
    def PlotRecError(self):
        plt.figure(8)
        plt.plot(self.network.reconstruction_error)
        plt.title("Mean Squared Error of SAILNet's Reconstruction with 25000 Iterations and STDP Learning Rule")
        plt.savefig(self.directory + '/Images/Rec_Error.png')
    
    def PlotInhibitHist(self):
        plt.figure(9)
        W_flat = np.ravel(self.network.W) #Flattens array
        zeros = np.nonzero(W_flat == 0) #Locates zeros
        W_flat = np.delete(W_flat, zeros) #Deletes Zeros
        W_flat = np.log(W_flat)/np.log(10)
        num, bin_edges = np.histogram(W_flat, bins = 10000, density = True)
        num = np.append(np.array([0]),num)
        plt.xlim([-6,2])
        plt.plot(bin_edges,num,'o')
        plt.xlabel("Inhibitory Connection Strength")
        plt.ylabel("PDF log(connection strength)")
        plt.title("Histogram of Inhibitory Connection Strengths for 25000 Iterations and STDP Learning Rule")
        plt.savefig(self.directory + '/Images/InhibitHist.png')
        
    def PlotInh_vs_RF(self):
        plt.figure(10)
        RF_overlap = self.network.Q.T.dot(self.network.Q)
        RF_overlap = np.ravel(RF_overlap)
        W_flat = np.ravel(self.network.W) #Flattens array
        zeros = np.nonzero(W_flat == 0) #Locates zeros
        W_flat = np.delete(W_flat, zeros) #Deletes Zeros
        RF_overlap = np.delete(RF_overlap, zeros) #Deletes Zeros
        W_flat = np.abs(np.log(W_flat))
        plt.plot(W_flat, RF_overlap, '.')
        plt.xlabel("Inhibitory Connection Strength")
        plt.ylabel("RF Overlap (Dot product)")
        plt.savefig(self.directory + '/Images/Inhibitory_vs_RF.png')
        
    def Plot_Mag_W(self):
        plt.figure(11)
        plt.title('Magnitude of Lateral Weight Matrix W')
        plt.plot(self.network.mag_W)
        plt.savefig(self.directory + '/Images/Magnitude_W.png')
        
    def PlotAll(self):
        plt.figure(self.Plot_RF())
        #plt.figure(self.PlotdWstdp())
        plt.figure(self.PlotdW())
        #plt.figure(self.Plotstdp())
        #plt.figure(self.Plotcor())
        plt.figure(self.PlotCavg())
        plt.figure(self.PlotYavg())
        plt.figure(self.PlotTimeDep())
        plt.figure(self.PlotRecError())
        plt.figure(self.PlotInhibitHist())
        plt.figure(self.PlotInh_vs_RF())
        plt.figure(self.Plot_Mag_W())




