import cPickle, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils import tile_raster_images
from activity import Activity
from data import Data

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
        small_bs = self.network.parameters.batch_size        
        batch_size = 50000
        parameters = self.network.parameters

        data = Data(os.path.join(os.environ['DATA_PATH'],'vanhateren/whitened_images.h5'),
            1000,
            parameters.batch_size,
            parameters.N,
            start=35)     
            
        self.network.to_gpu()	

        activity = Activity(self.network)

        self.big_X = np.zeros((batch_size,parameters.N))
        self.big_Y = np.zeros((batch_size,parameters.M))

        for i in range(batch_size/small_bs):
            
            data.make_X(self.network) 
            if contrast != 1.:
                self.network.X.set_value(self.network.X.get_value()*contrast)
            activity.get_acts()
            
            self.big_X[i*small_bs:(i+1)*small_bs,:] = self.network.X.get_value()
            self.big_Y[i*small_bs:(i+1)*small_bs,:] = self.network.Y.get_value()
        
        self.network.to_cpu()
        self.network.Y = self.big_Y
        self.network.X = self.big_X
            
    def Plot_RF(self,network_Q = None,filenum = ''):
        if network_Q != None:
            Q = network_Q.get_value()
            filenum = str(filenum)
            function = ''
        else:
            Q = self.network.Q
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
        fig = plt.figure()
        plt.title('Experimental Receptive Fields')
        plt.imsave(self.directory + '/Images/RFs/Exp_RF.png', img, cmap=plt.cm.Greys)
        plt.close(fig)
    
    def PlotdW(self):
        fig = plt.figure()
        plt.plot(self.monitor.mag_dW)
        plt.title('Magnitude dW')
        plt.xlabel("Number of Trials")
        plt.savefig(self.directory + '/Images/Magnitude_dW.png')
        plt.close(fig)
        
    def PlotYavg(self):
        fig = plt.figure()
        plt.plot(self.monitor.y_bar)
        plt.title("Average Y")
        plt.xlabel("Number of Trials")
        plt.savefig(self.directory + '/Images/Yavg.png')
        plt.close(fig)

    
    def PlotCavg(self):
        fig = plt.figure()
        plt.plot(self.monitor.Cyy_bar)
        plt.title("AverageY^2")
        plt.savefig(self.directory + '/Images/Cavg.png')
        plt.close(fig)
        
    def PlotSNR(self):
        fig = plt.figure()
        plt.plot(self.monitor.SNR,'g')
        plt.title('Signal to Noise Ratio')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/SNR.png')
        plt.close(fig)

    def PlotSNR_Norm(self):
        fig = plt.figure()
        plt.plot(self.monitor.SNR_Norm,'b')
        plt.title('Signal to Noise Ratio')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/SNR_Norm.png')   
        plt.close(fig)
       
    def PlotQ(self):
        fig = plt.figure()
        plt.plot(self.monitor.Q_stats[:,0])
        plt.title('Q Mean')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/Q_mean.png')
        plt.close(fig)
        fig = plt.figure()
        plt.plot(self.monitor.Q_stats[:,1],'b')
        plt.title('Q Standard Deviation')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/Q_std.png')
        plt.close(fig)

    def PlotW(self):
        fig = plt.figure()
        plt.plot(self.monitor.W_stats[:,0],'g')
        plt.title('W Mean')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/W_mean.png')
        plt.close(fig)
        fig = plt.figure()
        plt.plot(self.monitor.W_stats[:,1],'b')
        plt.title('W Standard Deviation')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/W_std.png')
        plt.close(fig)
        
    def PlotTheta(self):
        fig = plt.figure()
        plt.plot(self.monitor.theta_stats[:,0],'g')
        plt.title('Theta Mean')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/theta_mean.png')
        plt.close(fig)
        fig = plt.figure()
        plt.plot(self.monitor.theta_stats[:,1],'b')
        plt.title('Theta Standard Deviation')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/theta_std.png')
        plt.close(fig)

    def PlotX_rec(self):
        fig = plt.figure()
        plt.plot(self.monitor.X_rec_stats[:,0],'g')
        plt.title('Reconstructed Image (Y*Q) Mean')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/X_rec_mean.png')
        plt.close(fig)
        fig = plt.figure()
        plt.plot(self.monitor.X_rec_stats[:,1],'b')
        plt.title('Reconstructed Image (Y*Q) Standard Deviation')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/X_rec_std.png')
        plt.close(fig)

    def PlotX(self):
        fig = plt.figure()
        plt.plot(self.monitor.X_norm_bar,'b')
        plt.title('Image Norm Mean')
        plt.xlabel('Number of Trials')
        plt.savefig(self.directory + '/Images/X_norm_bar.png')
        plt.close(fig)
    
    def PlotInhibitHist(self):
        fig = plt.figure()
        W_flat = np.ravel(self.network.W) #Flattens array
        zeros = np.nonzero(W_flat == 0) #Locates zeros
        W_flat = np.delete(W_flat, zeros) #Deletes Zeros
        W_flat = np.log(W_flat)/np.log(10)
        num, bin_edges = np.histogram(W_flat,range = (-6,2), bins = 100, density = True)
        bin_edges = bin_edges[1:]
        bin_edges = 10**bin_edges
        plt.semilogx(bin_edges,num,'o')
        plt.ylim(0,0.9)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel("Inhibitory Connection Strength")
        plt.ylabel("PDF log(connection strength)")
        plt.savefig(self.directory + '/Images/InhibitHist.png')
        plt.close(fig)
        
    def PlotInh_vs_RF(self):
        fig = plt.figure()
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
        plt.close(fig)
        
    def Plot_Rate_Hist(self):
        fig = plt.figure()
        rates = np.mean(self.network.Y,axis = 0)
        num, bin_edges = np.histogram(rates,range = (0.025,0.08), bins = 50)
        bin_edges = bin_edges[1:]
        plt.plot(bin_edges,num,'o')
        #lt.ylim(0,100)
        #plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel("Mean Firing Rate")
        plt.ylabel("Number of Cells")
        plt.savefig(self.directory + '/Images/RateHist.png') 
        plt.close(fig)
     
    def Plot_Rate_Hist_LC(self):
        fig = plt.figure()
        self.validation_data(1/3.)        
        rates = np.mean(self.network.Y,axis = 0)
        num, bin_edges = np.histogram(rates,range = (0.025,0.08), bins = 50)
        bin_edges = bin_edges[1:]
        plt.plot(bin_edges,num,'o')
        #plt.ylim(0,100)
        #plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel("Mean Firing Rate")
        plt.ylabel("Number of Cells")
        plt.savefig(self.directory + '/Images/RateHistLC.png') 
        plt.close(fig)

    def Plot_Rate_Corr(self):
        fig = plt.figure()
        Y = self.network.Y
        corrcoef = np.corrcoef(Y,rowvar = 0)
        corrcoef = corrcoef - np.diag(np.diag(corrcoef))
        corrcoef = np.ravel(corrcoef) #Flattens array
        plt.hist(corrcoef,bins = 50,range = (-0.05,0.05),normed= True)
        #plt.ylim(0,300)
        #plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel("Rate Correlation")
        plt.ylabel("PDF")
        plt.savefig(self.directory + '/Images/RateCorrHist.png') 
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
        self.Plot_RF()
        self.PlotCavg()
        self.PlotYavg()
        self.PlotSNR()
        self.PlotSNR_Norm()
        self.PlotX_rec()
        self.PlotQ()
        self.PlotW()
        self.PlotTheta()
        self.PlotX()
        self.PlotInhibitHist()
        self.PlotInh_vs_RF()
        self.validation_data()
        self.Plot_EXP_RF()
        self.Plot_Rate_Hist()
        self.Plot_Rate_Corr()
        self.Plot_Rate_Hist_LC()


if __name__ == "__main__":
    directory = sys.argv[1]
    plotter = Plot(directory)
    plotter.load_network()
    print(plotter.find_last_spike())
    #plotter.PlotAll()
