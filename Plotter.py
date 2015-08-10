import cPickle, sys
import matplotlib.pyplot as plt
import numpy as np
from utils import tile_raster_images
import os
from Activity import Activity_gpu as Activity
from Data import Data
from Parameters import Parameters

class Plot():
    
    def __init__(self, directory):
	self.parameters = Parameters(os.path.join(directory,'parameters.txt'))
        self.directory = directory
        if os.path.exists(self.directory+'/Images')==False:       
            os.makedirs(self.directory+'/Images')
            os.makedirs(self.directory+'/Images/RFs')
            
    def load_network(self):
        self.fileName = self.directory + '/data.pkl'
        with open(self.fileName,'rb') as f:
            self.network, self.learning_rule, self.monitor = cPickle.load(f)
            
    def Plot_RF(self,network_Q = None,filenum = ''):
        if network_Q != None:
            Q = network_Q.get_value()
            filenum = str(filenum)
        else:
            Q = self.network.Q
            
        im_size, num_dict = Q.shape

        side = int(np.round(np.sqrt(im_size)))
        OC = num_dict/im_size

        img = tile_raster_images(Q.T, img_shape = (side,side),
                                 tile_shape = (2*side,side*OC/2), tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=True, output_pixel_vals=True)
        plt.figure()
        plt.title('Receptive Fields' + filenum)
        plt.imsave(self.directory + '/Images/RFs/Receptive_Fields'+str(self.parameters.function)+filenum+'.png', img, cmap=plt.cm.Greys)
    
    def validation_data(self):        
        self.network.parameters.batch_size = 10000
        parameters = self.network.parameters
        data = Data('/home/jesse/Development/data/vanhateren/whitened_images.h5',
            1000,
            parameters.batch_size,
            parameters.N,
            start=35)     
       
	self.network.to_gpu()	
	 
        data.make_X(self.network) 
        activity = Activity()
        activity.get_acts(self.network)

	self.network.to_cpu()
        
    def Plot_EXP_RF(self):
        Exp_RF = self.network.X.T.dot(self.network.Y)
        
        spike_sum = np.sum(self.network.Y,axis = 0,dtype='f')
        Exp_RF = Exp_RF.dot(np.diag(1/spike_sum))

        im_size, num_dict = Exp_RF.shape

        side = int(np.round(np.sqrt(im_size)))
        OC = num_dict/im_size

        img = tile_raster_images(Exp_RF.T, img_shape = (side,side),
                                 tile_shape = (2*side,side*OC/2), tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=True, output_pixel_vals=True)
        plt.figure()
        plt.title('Experimental Receptive Fields')
        plt.imsave(self.directory + '/Images/RFs/Exp_RF.png', img, cmap=plt.cm.Greys)
    
    def Plot_Rate_Hist(self):
	spike_sum = np.sum(self.network.Y,axis = 0)
	rates = spike_sum/len(self.network.Y[:,1])
	num, bin_edges = np.histogram(rates, range = (0.028,0.08),bins = 50)
	num = np.append(np.array([0]),num)
        bin_edges = 10**bin_edges
        plt.plot(bin_edges,num,'o')
        plt.ylim(0,100)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel("Mean Firing Rate")
        plt.ylabel("Number of Cells")
        plt.savefig(self.directory + '/Images/RateHist.png') 
     
    def Plot_Rate_Hist_LC(self):
	
	spike_sum = np.sum(self.network.Y,axis = 0)
	rates = spike_sum/len(self.network.Y[:,1])
	num, bin_edges = np.histogram(rates, range = (0.028,0.08),bins = 50)
	num = np.append(np.array([0]),num)
        bin_edges = 10**bin_edges
        plt.plot(bin_edges,num,'o')
        plt.ylim(0,100)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel("Mean Firing Rate")
        plt.ylabel("Number of Cells")
        plt.savefig(self.directory + '/Images/RateHist.png') 

    def Plot_Rate_Corr(self):
	corrcoef = np.array([])
	Y = self.network.Y
	num_neurons = len(self.network.parameters.M)
	for i in range(num_neurons):
	    for j in range(num_neurons):
		if i <j:
		    corrcoef = corrcoef.append(np.corrcoef(Y[:,i],

    def PlotdW(self):
        plt.figure(18)
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
        
        plt.savefig(self.directory + '/Images/Cavg.png')
        
    def PlotSNR(self):
        plt.figure(6)
        plt.plot(self.monitor.SNR,'g')
        plt.title('Signal to Noise Ratio')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/SNR.png')

    def PlotSNR_Norm(self):
        plt.figure(22)
        plt.plot(self.monitor.SNR_Norm,'b')
        plt.title('Signal to Noise Ratio')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/SNR_Norm.png')        
       
    def PlotQ(self):
        plt.figure(7)
        plt.plot(self.monitor.Q_stats[:,0])
        plt.title('Q Mean')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/Q_mean.png')
        plt.figure(17)
        plt.plot(self.monitor.Q_stats[:,1],'b')
        plt.title('Q Standard Deviation')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/Q_std.png')

    def PlotW(self):
        plt.figure(8)
        plt.plot(self.monitor.W_stats[:,0],'g')
        plt.title('W Mean')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/W_mean.png')
        plt.figure(19)
        plt.plot(self.monitor.W_stats[:,1],'b')
        plt.title('W Standard Deviation')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/W_std.png')
        
    def PlotTheta(self):
        plt.figure(9)
        plt.plot(self.monitor.theta_stats[:,0],'g')
        plt.title('Theta Mean')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/theta_mean.png')
        plt.figure(15)
        plt.plot(self.monitor.theta_stats[:,1],'b')
        plt.title('Theta Standard Deviation')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/theta_std.png')

    def PlotX_rec(self):
        plt.figure(10)
        plt.plot(self.monitor.X_rec_stats[:,0],'g')
        plt.title('Reconstructed Image (Y*Q) Mean')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/X_rec_mean.png')
        plt.figure(14)
        plt.plot(self.monitor.X_rec_stats[:,1],'b')
        plt.title('Reconstructed Image (Y*Q) Standard Deviation')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/X_rec_std.png')

    def PlotX(self):
        plt.figure(16)
        plt.plot(self.monitor.X_norm_bar,'b')
        plt.title('Image Norm Mean')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/X_norm_bar.png')
 
    
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
        plt.figure(8)
        plt.plot(self.network.reconstruction_error)
        plt.title("Mean Squared Error of SAILNet's Reconstruction with 25000 Iterations and STDP Learning Rule")
        plt.savefig(self.directory + '/Images/Rec_Error.png')
    
    
    def PlotInhibitHist(self):
        plt.figure(11)
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
        plt.figure(12)
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
        plt.figure(13)
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
        
    def find_last_spike(self):
        latest_spike = np.array([])
        spikes = self.network.spike_train
        for batch in range(len(spikes[:,0,0])):
            S = spikes[batch,:,:]
            print batch
            R,C = np.nonzero(S)
            N,I = np.unique(R,return_index =True)
            latest_spike = np.append(latest_spike,max(C[I]))
        return latest_spike
        
    def PlotAll(self):
        plt.figure(self.Plot_RF())
        plt.figure(self.PlotdW())
        plt.figure(self.PlotCavg())
        plt.figure(self.PlotYavg())
        plt.figure(self.PlotSNR())
        plt.figure(self.PlotSNR_Norm())
	plt.figure(self.PlotX_rec())
        plt.figure(self.PlotQ())
        plt.figure(self.PlotW())
        plt.figure(self.PlotTheta())
	plt.figure(self.PlotX())
        #plt.figure(self.Plotcor())
        #plt.figure(self.PlotTimeDep())
        #plt.figure(self.PlotRecError())
        plt.figure(self.PlotInhibitHist())
        plt.figure(self.PlotInh_vs_RF())
        #plt.figure(self.Plot_Mag_W())
        plt.figure(self.Plot_Exp_RF())


if __name__ == "__main__":
    directory = sys.argv[1]
    plotter = Plot(directory)
    plotter.load_network()
    plotter.PlotAll()
