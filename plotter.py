import cPickle, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils import tile_raster_images
from activity import Activity
from data import Data
from matplotlib.backends.backend_pdf import PdfPages


class Plot():
    
    def __init__(self, directory):
        self.directory = directory
        if os.path.exists(self.directory+'/Images')==False:       
            os.makedirs(self.directory+'/Images')
            os.makedirs(self.directory+'/Images/RFs')
            
    def load_network(self):
        self.fileName = self.directory + '/data.pkl'
        with open(self.fileName,'rb') as f:
            self.network, self.monitor, _ = cPickle.load(f)
        self.parameters = self.network.parameters
            
    def validation_data(self,contrast = 1.):
        self.network.parameters.batch_size = 1000
        orig = self.network.parameters.time_data
        self.network.parameters.time = False
        small_bs = self.network.parameters.batch_size        
        batch_size = 50000
        parameters = self.network.parameters
        
        if parameters.time_data:
            data = Time_Data(os.path.join(os.environ['DATA_PATH'],'vanhateren/whitened_images.h5'),
            1000,
            parameters.batch_size,
            parameters.N,
            parameters.num_frames,
            start=35)     
        else:
            data = Static_Data(os.path.join(os.environ['DATA_PATH'],'vanhateren/whitened_images.h5'),
            1000,
            parameters.batch_size,
            parameters.N,
            start=35)    

        self.network.to_gpu()	

        activity = Activity(self.network)

        self.big_X = np.zeros((batch_size,parameters.N))
        self.big_Y = ()
        
        for layer in range(self.network.n_layers):
            self.big_Y += (np.zeros((batch_size,parameters.M[layer])),)

        for i in range(batch_size/small_bs):
            
            data.make_X(self.network) 
            if contrast != 1.:
                self.network.X.set_value(self.network.X.get_value()*contrast)
            activity.get_acts()
            
            self.big_X[i*small_bs:(i+1)*small_bs,:] = self.network.X.get_value()
            for layer in range(self.network.n_layers):
                self.big_Y[layer][i*small_bs:(i+1)*small_bs,:] = self.network.Y[layer].get_value()
        
        self.network.to_cpu()
        self.network.Y = self.big_Y
        self.network.X = self.big_X
        self.network.parameters.time_data = orig
            
    def Plot_RF(self,network_Q = None,layer = 0,filenum = ''):
                
        if network_Q != None:
            Q = network_Q[layer].get_value()
            filenum = str(filenum)
            function = ''
        else:
            Q = self.network.Q[layer]
            function = self.network.parameters.function
            
        im_size, num_dict = Q.shape

        side = int(np.round(np.sqrt(im_size)))
        OC = num_dict/im_size

        img = tile_raster_images(Q.T, img_shape = (side,side),
                                 tile_shape = (2*side,side*OC/2), tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=True, output_pixel_vals=True)
        fig = plt.figure()
        plt.title('Receptive Fields' + filenum)
        plt.imsave(self.directory + '/Images/RFs/Receptive_Fields'+function+filenum+'.png', img, cmap=plt.cm.Greys)
        plt.close(fig)
        
    def Plot_EXP_RF(self,layer=0):
        Exp_RF = self.network.X.T.dot(self.network.Y[layer])
        
        spike_sum = np.sum(self.network.Y[layer],axis = 0,dtype='f')
        Exp_RF = Exp_RF.dot(np.diag(1/spike_sum))

        im_size, num_dict = Exp_RF.shape

        side = int(np.round(np.sqrt(im_size)))
        OC = num_dict/im_size

        img = tile_raster_images(Exp_RF.T, img_shape = (side,side),
                                 tile_shape = (2*side,side*OC/2), tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=True, output_pixel_vals=True)
        fig = plt.figure()
        plt.title('Experimental Receptive Fields')
        plt.imsave(self.directory + '/Images/RFs/Exp_RF.png', img, cmap=plt.cm.Greys)
        plt.close(fig)
        
    def plot_training_values(self, layer_num, channel):
        fig = plt.figure()
        plt.plot(self.monitor.channels[channel][layer_num])
        plt.title(channel+' Layer '+str(layer_num))
        plt.xlabel("Number of Trials")
        self.pp.savefig(fig)
        plt.close(fig)
        
    def plot_training_mean_std(self, layer_num, channel):
        fig = plt.figure()
        mean = self.monitor.channels[channel][layer_num][:,0]
        std = self.monitor.channels[channel][layer_num][:,1]
        plt.plot(mean)
        plt.fill_between(range(len(mean)),mean-std, mean+std,
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.title(channel+' Layer '+str(layer_num))
        plt.xlabel('Number of Trials')
        self.pp.savefig(fig)
        plt.close(fig)
    
    def PlotdW(self, layer_num):
        fig = plt.figure()
        plt.plot(self.monitor.mag_dW[layer_num])
        plt.title('Magnitude dW')
        plt.xlabel("Number of Trials")
        self.pp.savefig()
        plt.close(fig)
        
    def PlotYavg(self, layer_num):
        fig = plt.figure()
        plt.plot(self.monitor.y_bar[layer_num])
        plt.title("Average Y")
        plt.xlabel("Number of Trials")
        self.pp.savefig()
        plt.close(fig)[layer_num]

    
    def PlotCavg(self, layer_num):
        fig = plt.figure()
        plt.plot(self.monitor.Cyy_bar[layer_num])
        plt.title("AverageY^2")
        self.pp.savefig()
        plt.close(fig)
        
    def PlotSNR(self, layer_num):
        fig = plt.figure()
        plt.plot(self.monitor.SNR[layer_num],'g')
        plt.title('Signal to Noise Ratio')
        plt.xlabel('Number of Trials')
        self.pp.savefig()
        plt.close(fig)

    def PlotSNR_Norm(self,layer_num):
        fig = plt.figure()
        plt.plot(self.monitor.SNR_Norm[layer_num],'b')
        plt.title('Normalized Signal to Noise Ratio')
        plt.xlabel('Number of Trials')
        self.pp.savefig()
        plt.close(fig)
       
    def PlotQ(self,layer_num):
        fig = plt.figure()
        mean = self.monitor.Q_stats[layer_num][:,0]
        std = self.monitor.Q_stats[layer_num][:,1]
        plt.plot(mean)
        plt.fill_between(range(len(mean)),mean-std, mean+std,
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.title('Q Mean')
        plt.xlabel('Number of Trials')
        self.pp.savefig()
        plt.close(fig)

    def PlotW(self,layer_num):
        fig = plt.figure()
        mean = self.monitor.W_stats[layer_num][:,0]
        std = self.monitor.W_stats[layer_num][:,1]
        plt.plot(mean)
        plt.fill_between(range(len(mean)),mean-std, mean+std,
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.title('W Mean')
        plt.xlabel('Number of Trials')
        self.pp.savefig()
        plt.close(fig)
        
    def PlotTheta(self,layer_num):
        fig = plt.figure()
        mean = self.monitor.theta_stats[layer_num][:,0]
        std = self.monitor.theta_stats[layer_num][:,1]
        plt.plot(mean)
        plt.fill_between(range(len(mean)),mean-std, mean+std,
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.title('Theta Mean')
        plt.xlabel('Number of Trials')
        self.pp.savefig()
        plt.close(fig)

    def PlotX_rec(self,layer_num):
        fig = plt.figure()
        mean = self.monitor.X_rec_stats[layer_num][:,0]
        std = self.monitor.X_rec_stats[layer_num][:,1]
        plt.plot(mean)
        plt.fill_between(range(len(mean)),mean-std, mean+std,
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.title('Reconstructed Image (Y*Q) Mean')
        plt.xlabel('Number of Trials')
        self.pp.savefig()
        plt.close(fig)

    def PlotX(self,layer_num):
        fig = plt.figure()
        plt.plot(self.monitor.X_norm_bar[layer_num],'b')
        plt.title('Image Norm Mean')
        plt.xlabel('Number of Trials')
        self.pp.savefig()
        plt.close(fig)
    
    def PlotInhibitHistLogX(self,layer=0):
        fig = plt.figure()
        W_flat = np.ravel(self.network.W[layer]) #Flattens array
        zeros = np.nonzero(W_flat == 0) #Locates zeros
        W_flat = np.delete(W_flat, zeros) #Deletes Zeros
        W_flat = np.log10(W_flat)
        num, bin_edges = np.histogram(W_flat, bins = 100, density = True)
        bin_edges = bin_edges[1:]
        bin_edges = 10**bin_edges
        plt.semilogx(bin_edges,num,'o')
        plt.ylim(0,0.9)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('Inhibitory Strength Histogram Log X')        
        plt.xlabel("log(Inhibitory Connection Strength)")
        plt.ylabel("PDF log(connection strength)")
        self.pp.savefig(fig)
        plt.close(fig)
        
    def PlotInhibitHistLogY(self,layer=0):
        fig = plt.figure()
        W_flat = np.ravel(self.network.W[layer]) #Flattens array
        zeros = np.nonzero(W_flat == 0) #Locates zeros
        W_flat = np.delete(W_flat, zeros) #Deletes Zeros
        num, bin_edges = np.histogram(W_flat,range=(0.00001,3), bins = 100, density = True)
        bin_edges = bin_edges[1:]
        plt.semilogy(bin_edges,num,'o')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('Inhibitory Strength Histogram Log Y')        
        plt.xlabel("Inhibitory Connection Strength")
        plt.ylabel("log (PDF connection strength)")
        self.pp.savefig(fig)
        plt.close(fig)
        
    def PlotInhibitHist(self,layer=0):
        fig = plt.figure()
        W = self.network.W[layer] - np.diag(np.diag(self.network.W[layer]))
        W_flat = np.ravel(W) #Flattens array
        zeros = np.nonzero(W_flat == 0) #Locates zeros
        W_flat = np.delete(W_flat, zeros) #Deletes Zeros
        num, bin_edges = np.histogram(W_flat,range=(0.00001,3), bins = 500,density=True)
        bin_edges = bin_edges[1:]
        plt.plot(bin_edges,num,'o')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('Inhibitory Strength Histogram')        
        plt.xlabel("Inhibitory Connection Strength")
        plt.ylabel("PDF connection strength")
        self.pp.savefig(fig)
        plt.close(fig)
        
    def PlotInh_vs_RF(self,layer=0):
        fig = plt.figure()
        Q = self.network.Q[layer]
        RF_overlap = Q.T.dot(Q)
        pairs = np.random.randint(0,self.network.parameters.M[layer],(5000,2))
        RF_sample = np.array([])
        W_sample = np.array([])
        for pair in pairs:
            Overlap = RF_overlap[pair[0]][pair[1]]
            RF_sample = np.append(RF_sample, np.array([Overlap]))
            w1 = self.network.W[layer][pair[0]][pair[1]]
            W_sample = np.append(W_sample,np.array([w1]))
        #plt.xlim(10**-3,10**1.5)
        plt.semilogx(W_sample, RF_sample, '.')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('Inhibitory Connection Str vs RF Overlap')
        plt.xlabel("Inhibitory Connection Strength")
        #plt.ylim(-0.7,0.7)
        plt.ylabel("RF Overlap (Dot product)")
        self.pp.savefig(fig)
        plt.close(fig)
        
    def Plot_Rate_Hist(self,layer=0):
        fig = plt.figure()
        rates = np.mean(self.network.Y[layer],axis = 0)
        num, bin_edges = np.histogram(rates, bins = 50)
        bin_edges = bin_edges[1:]
        plt.plot(bin_edges,num,'o')
        #lt.ylim(0,100)
        #plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('Rate Histogram')
        plt.xlabel("Mean Firing Rate")
        plt.ylabel("Number of Cells")
        self.pp.savefig(fig)
        plt.close(fig)
     
    def Plot_Rate_Hist_LC(self,layer=0):
        fig = plt.figure()
        self.validation_data(1/3.)        
        rates = np.mean(self.network.Y[layer],axis = 0)
        num, bin_edges = np.histogram(rates, bins = 50)
        bin_edges = bin_edges[1:]
        plt.plot(bin_edges,num,'o')
        #plt.ylim(0,100)
        #plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('Low Contrast Rate Histogram')
        plt.xlabel("Mean Firing Rate")
        plt.ylabel("Number of Cells")
        self.pp.savefig(fig)
        plt.close(fig)

    def Plot_Rate_Corr(self,layer=0):
        fig = plt.figure()
        Y = self.network.Y[layer]
        corrcoef = np.corrcoef(Y,rowvar = 0)
        corrcoef = corrcoef - np.diag(np.diag(corrcoef))
        corrcoef = np.ravel(corrcoef) #Flattens array
        plt.hist(corrcoef,bins = 50,normed= True)
        #plt.ylim(0,300)
        #plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('Correlation PDF')
        plt.xlabel("Rate Correlation")
        plt.ylabel("PDF")
        self.pp.savefig(fig)
        plt.close(fig)
        
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
            R,C = np.nonzero(S)
            N,I = np.unique(R,return_index =True)
            latest_spike = np.append(latest_spike,max(C[I]))
        return latest_spike
        
    def PlotAll(self):
        with PdfPages(self.directory+'/Images/plots.pdf') as self.pp:
            self.validation_data()
            self.Plot_RF()
            for layer in range(self.network.n_layers):
                for channel in self.monitor.training_values:
                    self.plot_training_values(layer, channel)
                for channel in self.monitor.training_mean_std:
                    self.plot_training_mean_std(layer, channel)

                self.PlotInhibitHistLogX(layer) 
                self.PlotInhibitHistLogY(layer)
                self.PlotInhibitHist(layer)
                self.PlotInh_vs_RF(layer)
                self.Plot_EXP_RF(layer)
                self.Plot_Rate_Hist(layer)
                self.Plot_Rate_Corr(layer)
                self.Plot_Rate_Hist_LC(layer)


if __name__ == "__main__":
    directory = sys.argv[1]
    plotter = Plot(directory)
    plotter.load_network()
    plotter.PlotAll()
