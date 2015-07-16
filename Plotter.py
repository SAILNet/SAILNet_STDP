import cPickle
import matplotlib.pyplot as plt
import numpy as np
from utils import tile_raster_images
import os


class Plot():
    
    def __init__(self,fileName, directory):
        self.fileName=fileName
        self.directory = directory
        if os.path.exists(self.directory+'/Images')==False:       
            os.makedirs(self.directory+'/Images')
        with open(self.fileName,'rb') as f:
            self.network, self.learning_rule, self.monitor = cPickle.load(f)
            
    def Plot_RF(self):
        im_size, num_dict = self.network.Q.shape

        side = int(np.round(np.sqrt(im_size)))
        OC = num_dict/im_size
        print('j')

        img = tile_raster_images(self.network.Q.T, img_shape = (side,side),
                                 tile_shape = (2*side,side*OC/2), tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=True, output_pixel_vals=True)
        print('k')
        #plt.figure(0)
        print('f')
        plt.title('Receptive Fields')
        plt.imsave(self.directory + '/Images/Receptive_Fields.png', img, cmap=plt.cm.Greys)
    
    def PlotdW(self):
        #plt.figure(3)
        plt.plot(self.monitor.mag_dW)
        plt.title('Magnitude dW')
        plt.xlabel("Number of Trials")
        plt.savefig(self.directory + '/Images/Magnitude_dW.png')
        
    def PlotYavg(self):
        plt.figure(4)
        plt.plot(self.monitor.y_bar)
        plt.title("Average Y")
        plt.xlabel("Number of Trials")
        plt.savefig(self.directory + '/Images/Yavg.png')
    
    def PlotCavg(self):
        plt.figure(5)
        plt.plot(self.monitor.Cyy_bar)
        plt.title("AverageY^2")
        plt.xlabel("Number of Trials")
        plt.savefig(self.directory + '/Images/Cavg.png')
        
    def PlotSNR(self):
        plt.figure(6)
        plt.plot(self.monitor.SNR)
        plt.title('Signal to Noise Ratio')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/SNR.png')
    
    def Plotcor(self):
        plt.figure(6)
        plt.plot(self.correlation, color = "purple")
        plt.title('Correlation of dW and STDP')
        plt.xlabel("Number of Trials")
        plt.savefig(self.directory + '/Images/Correlation_dW_STDP.png')
    
    def PlotTimeDep(self):
        try:
            plt.figure(7)
            plt.plot(self.learning_rule.time_dep[25])
            plt.title(self.fileName[:len(self.fileName)-4] + 'Time Weighting Matrix')
            plt.savefig(self.directory + '/Images/Weighting_Matrix.png')
        except(AttributeError):
            pass
        
    def PlotRecError(self):
        #plt.figure(8)
        plt.plot(self.network.reconstruction_error)
        plt.title("Mean Squared Error of SAILNet's Reconstruction with 25000 Iterations and STDP Learning Rule")
        plt.savefig(self.directory + '/Images/Rec_Error.png')
    
    
    def PlotInhibitHist(self):
        plt.figure(9)
        W_flat = np.ravel(self.network.W) #Flattens array
        zeros = np.nonzero(W_flat == 0) #Locates zeros
        W_flat = np.delete(W_flat, zeros) #Deletes Zeros
        W_flat = np.log(W_flat)/np.log(10)
        num, bin_edges = np.histogram(W_flat,range = (-6,2), bins = 100, density = True)
        num = np.append(np.array([0]),num)
        bin_edges = 10**bin_edges
        plt.semilogx(bin_edges,num,'o')
        plt.ylim(0,0.9)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel("Inhibitory Connection Strength")
        plt.ylabel("PDF log(connection strength)")
        plt.savefig(self.directory + '/Images/InhibitHist.png')
        
    def PlotInh_vs_RF(self):
        plt.figure(10)
        RF_overlap = self.network.Q.T.dot(self.network.Q)
        pairs = np.random.randint(0,self.network.parameters.M,(5000,2))
        RF_sample = np.array([])
        W_sample = np.array([])
        for pair in pairs:
            Overlap = RF_overlap[pair[0]][pair[1]]
            RF_sample = np.append(RF_sample, np.array([Overlap]))
            w1 = self.network.W[pair[0]][pair[1]]
            #w2 = self.network.W[pair[1]][pair[0]]
            #w_avg = (w1+w2)/2
            W_sample = np.append(W_sample,np.array([w1]))
        #zeros = np.nonzero(W_sample == 0) #Locates zeros
        #W_sample = np.delete(W_sample, zeros) #Deletes Zeros
        #RF_sample = np.delete(RF_sample,zeros)
        #W_sample = np.log(W_sample)/np.log(10)
        plt.xlim(10**-3,10**1.5)
        plt.semilogx(W_sample, RF_sample, '.')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel("Inhibitory Connection Strength")
        #plt.ylim(-0.7,0.7)
        plt.ylabel("RF Overlap (Dot product)")
        plt.savefig(self.directory + '/Images/Inhibitory_vs_RF.png')
    
    def Plot_Mag_W(self):
        plt.figure(11)
        plt.title('Magnitude of Lateral Weight Matrix W')
        plt.plot(self.network.mag_W)
        plt.savefig(self.directory + '/Images/Magnitude_W.png')
        
    def RasterPlot(self):
        
        spikes = self.network.spike_train[5][:][:]
        
        check = np.nonzero(spikes)
        
        reducedSpikes = np.zeros([len(check[0]),50])
        
        neuron = 0
        for j in check[0]:
        
            reducedSpikes[neuron] = spikes[j]
            neuron += 1
        
        plt.figure()
        count1 = 0
        for neuron in(reducedSpikes):
            count =0
            for timestep in neuron:
                if timestep != 0:
                    plt.vlines(count, count1 +.5, count1 +1.4)            
                count += 1  
            count1 += 1
            
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel('time',**{'size':'25'})
        plt.ylabel('Neuron',**{'size':'25'})
        
        return reducedSpikes
        
        
    def PlotAll(self):
        plt.figure(self.Plot_RF())
        plt.figure(self.PlotdW())
        plt.figure(self.PlotCavg())
        plt.figure(self.PlotYavg())
        plt.figure(self.PlotSNR())
        #plt.figure(self.Plotcor())
        #plt.figure(self.PlotTimeDep())
        #plt.figure(self.PlotRecError())
        plt.figure(self.PlotInhibitHist())
        plt.figure(self.PlotInh_vs_RF())
        #plt.figure(self.Plot_Mag_W())




