import cPickle, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import network as nw
from utils import tile_raster_images
from activity import Activity
from data import Static_Data, Time_Data
from matplotlib.backends.backend_pdf import PdfPages


class Plot():
    
    def __init__(self, directory, seed=20150918):
        self.directory = directory
        if not os.path.exists(self.directory+'/Images'):       
            os.makedirs(self.directory+'/Images')
            os.makedirs(self.directory+'/Images/RFs')
        self.rng = np.random.RandomState(seed)
            
    def load_network(self):
        self.fileName = os.path.join(self.directory, 'data.pkl')
        with open(self.fileName,'rb') as f:
            self.network, self.monitor, _ = cPickle.load(f)
        self.parameters = self.network.parameters
            
    def validation_data(self, contrast=1., small_batch_size = 1000,large_batch_size = 50000):

        parameters = self.network.parameters        
        parameters.batch_size = small_batch_size
        orig_time_data = parameters.time_data
        orig_keep_spikes = parameters.keep_spikes
        #parameters.time_data = True
        #parameters.static_data_control = True
        parameters.keep_spikes = True
        if orig_keep_spikes == False:
            self.network.spike_train = ()
            nout = parameters.M
            for ii in np.arange(self.network.n_layers):
                out_dim = nout[ii]
                self.network.spike_train += (nw.make_shared((parameters.batch_size,
                                                  out_dim,
                                                  parameters.num_iterations)),)
                                                  
        small_bs = self.network.parameters.batch_size        
        batch_size = large_batch_size
        
        if parameters.time_data and not parameters.static_data_control:
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
        self.big_X = np.zeros((batch_size, parameters.N), dtype='float32')
        self.big_Y = ()
        
        for layer in range(self.network.n_layers):
            self.big_Y += (np.zeros((batch_size, parameters.M[layer]), dtype='float32'),)

        for ii in range(batch_size/small_bs):
            data.make_X(self.network) 
            if contrast != 1.:
                self.network.X.set_value(self.network.X.get_value() *
                                         np.array(contrast, dtype='float32'))
            activity.get_acts()
            
            self.big_X[ii*small_bs:(ii+1)*small_bs,:] = self.network.X.get_value()
            for layer in range(self.network.n_layers):
                self.big_Y[layer][ii*small_bs:(ii+1)*small_bs,:] = self.network.Y[layer].get_value()
        
        self.network.to_cpu()
        self.network.Y = self.big_Y
        self.network.X = self.big_X
        self.network.parameters.time_data = orig_time_data
        self.network.parameters.keep_spikes = orig_keep_spikes
            

    def frame_spike_correlation(self):
        small_bs = 250
        large_bs = 5000
        self.validation_data(1.,small_bs,large_bs)
        organized_spikes = self.network.Y.reshape((large_bs/(small_bs*20),20,small_bs,self.network.M))
        avg_distances = np.zeros((20,len(organized_spikes)))
        for index,saccade in enumerate(organized_spikes):
            for i in range(20):
                diff_spikes = saccade[i+1] - saccade[0]
                diff_spikes = np.linalg.norm(diff_spikes,axis = 1)
                avg_diff_spikes = np.mean(diff_spikes)
                avg_distances[i,index] = avg_diff_spikes
        avg_distances = np.mean(avg_distances,axis=1)
        plt.plot(avg_distances)
        plt.title("Spike Distance vs. Pixel Distance")
        plt.xlabel('Step Number')
        plt.ylabel('Spike Difference Norm') 
        self.pp.savefig()
        plt.close()


    def Plot_RF(self, network_Q=None, layer=0, filenum=''):
        if network_Q != None:
            Q = network_Q[layer].get_value()
            filenum = str(filenum)
            function = ''
        else:
            Q = self.network.Q[layer]
            function = self.network.parameters.function
        im_size, num_dict = Q.shape

        side = int(np.round(np.sqrt(im_size)))
        im_rows = int(np.sqrt(num_dict))
        if im_rows**2 < num_dict:
            im_cols = im_rows+1
        else:
            im_cols = im_rows
        OC = num_dict/im_size

        img = tile_raster_images(Q.T, img_shape=(side, side),
                                 tile_shape=(im_rows, im_cols), tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=True, output_pixel_vals=True)
        fig = plt.figure()
        plt.title('Receptive Fields' + filenum)
        plt.axis('off')
        plt.imsave(self.directory + '/Images/RFs/Receptive_Fields'+function+filenum+'.png', img, cmap=plt.cm.Greys)
        plt.close(fig)
        
    def Plot_EXP_RF(self, layer=0):
        Exp_RF = self.network.X.T.dot(self.network.Y[layer])
        
        spike_sum = np.sum(self.network.Y[layer],axis = 0,dtype='f')
        Exp_RF = Exp_RF.dot(np.diag(1/spike_sum))

        im_size, num_dict = Exp_RF.shape

        side = int(np.round(np.sqrt(im_size)))
        im_rows = int(np.sqrt(num_dict))
        if im_rows**2 < num_dict:
            im_cols = im_rows+1
        else:
            im_cols = im_rows
        OC = num_dict/im_size

        img = tile_raster_images(Exp_RF.T, img_shape=(side,side),
                                 tile_shape=(im_rows, im_cols), tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=True, output_pixel_vals=True)
        fig = plt.figure()
        plt.title('Experimental Receptive Fields Layer '+str(layer))
        plt.axis('off')
        plt.imsave(self.directory + '/Images/RFs/Exp_RF_'+str(layer)+'.png', img, cmap=plt.cm.Greys)
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
    
    def PlotInhibitHistLogX(self,layer=0):
        W_flat = np.ravel(self.network.W[layer]) #Flattens array
        W_flat = W_flat[W_flat > 0.]
        W_flat = np.log10(W_flat)
        num, bin_edges = np.histogram(W_flat, bins=100, density=True)
        bin_edges = bin_edges[1:]
        bin_edges = 10**bin_edges
        if num.max() > 0.:
            fig = plt.figure()
            plt.semilogx(bin_edges, num, 'o')
            plt.ylim(0,0.9)
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.title('Inhibitory Strength Histogram Log X')        
            plt.xlabel("log(Inhibitory Connection Strength)")
            plt.ylabel("PDF log(connection strength)")
            self.pp.savefig(fig)
            plt.close(fig)
        
    def PlotInhibitHistLogY(self,layer=0):
        W_flat = np.ravel(self.network.W[layer]) #Flattens array
        W_flat = W_flat[W_flat > 0.]
        num, bin_edges = np.histogram(W_flat, bins=100, density=True)
        bin_edges = bin_edges[1:]
        if num.max() > 0.:
            fig = plt.figure()
            plt.semilogy(bin_edges, num, 'o')
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.title('Inhibitory Strength Histogram Log Y')        
            plt.xlabel("Inhibitory Connection Strength")
            plt.ylabel("log (PDF connection strength)")
            self.pp.savefig(fig)
            plt.close(fig)
        
    def PlotInhibitHist(self,layer=0):
        W_flat = np.ravel(self.network.W[layer]) #Flattens array
        W_flat = W_flat[W_flat > 0.]
        num, bin_edges = np.histogram(W_flat, bins=100, density=True)
        bin_edges = bin_edges[1:]
        if num.max() > 0.:
            fig = plt.figure()
            plt.plot(bin_edges, num, 'o')
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.title('Inhibitory Strength Histogram')        
            plt.xlabel("Inhibitory Connection Strength")
            plt.ylabel("PDF connection strength")
            self.pp.savefig(fig)
            plt.close(fig)
        
    def PlotInh_vs_RF(self, layer=0):
        Q = self.network.Q[layer]
        W = self.network.W[layer]
        n_neurons = Q.shape[1]
        RF_overlap = Q.T.dot(Q)
        pairs = 5000
        RF_sample = np.array([])
        W_sample = np.array([])
        for ii in range(pairs):
            pair = self.rng.permutation(n_neurons)[:2]
            RFO_ij = RF_overlap[pair[0], pair[1]]
            W_ij = W[pair[0], pair[1]]
            if W_ij > 0.:
                RF_sample = np.append(RF_sample,RFO_ij)
                W_sample = np.append(W_sample,W_ij)
        if W_sample.size > 0 and not np.any(np.isnan(RF_sample)):
            fig = plt.figure()
            #plt.xlim(10**-3,10**1.5)
            plt.semilogx(W_sample, RF_sample, '.')
            #plt.gcf().subplots_adjust(bottom=0.15)
            plt.title('Inhibitory Connection Str vs RF Overlap')
            plt.xlabel("Log Inhibitory Connection Strength")
            #plt.ylim(-0.7,0.7)
            plt.ylabel("RF Overlap (Dot product)")
            self.pp.savefig(fig)
            plt.close(fig)
        
    def Plot_Rate_Hist(self,layer=0):
        rates = np.mean(self.network.Y[layer],axis = 0)
        num, bin_edges = np.histogram(rates, bins = 50)
        bin_edges = bin_edges[1:]
        fig = plt.figure()
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

    def Plot_Rate_Corr(self, layer=0):
        Y = self.network.Y[layer]
        n_neurons = Y.shape[1]
        corrcoef = np.corrcoef(Y,rowvar = 0)
        corrcoef_flat = np.array([])
        for ii in range(n_neurons-1):
            corrcoef_flat = np.append(corrcoef_flat,corrcoef[ii,ii+1:])
        corrcoef_flat = corrcoef_flat[np.logical_not(np.isnan(corrcoef_flat))]
        if corrcoef_flat.size > 0.:
            fig = plt.figure()
            plt.hist(corrcoef_flat, 50, normed= True)
            #plt.ylim(0,300)
            #plt.gcf().subplots_adjust(bottom=0.15)
            plt.title('Correlation PDF')
            plt.xlabel("Rate Correlation")
            plt.ylabel("PDF")
            self.pp.savefig(fig)
            plt.close(fig)
    
    def Plot_Rate_vs_Time(self,layer):
        spike_train = self.network.spike_train[layer]        
        rates = spike_train.mean(0).mean(0)
        fig = plt.figure()
        plt.plot(rates)
        plt.title('Mean Firing Rates vs Time')
        plt.ylabel('Mean Firing Rates')
        plt.xlabel('Number of Iterations')
        self.pp.savefig(fig)
        plt.close(fig)
        
    def Plot_Raster(self,layer):
        spike_train = self.network.spike_train[layer]
        num_on = 0
        idx = 0
        
        for ii in xrange(spike_train.shape[0]):
            if np.count_nonzero(spike_train[ii].sum(axis=1)) > num_on:
                idx = ii
        spikes = spike_train[idx]
        spike_sum = spikes.sum(axis=1)
        num_on = np.count_nonzero(spike_sum)
        
        max_args = np.argsort(spike_sum)[::-1]
        max_args = max_args[:num_on]
        
        if num_on > 0:
            rand_args = self.rng.permutation(num_on)[:min(num_on, 10)]
            spikes_subset = spikes[rand_args]

            fig = plt.figure()
            colors = np.array(matplotlib.colors.cnames.keys())[[0,41,42,53,70,118,89,97,102,83]]
            for i,neuron in enumerate(spikes_subset):
                neuron = np.nonzero(neuron)[0]
                plt.vlines(neuron, i +.5, i +1.2,colors[i])            
            plt.ylim(.5,len(spikes_subset)+0.5)         
            
            plt.title('Raster Plot Layer '+layer,{'fontsize':'25'})
            plt.xlabel('Time')
            plt.ylabel('Neuron')
            self.pp.savefig(fig)
            plt.close(fig)

    def find_last_spike(self):
        latest_spike = np.array([])
        spikes = self.network.spike_train
        for batch in range(len(spikes[:,0,0])):
            S = spikes[batch,:,:]
            R,C = np.nonzero(S)
            N,I = np.unique(R,return_index =True)
            latest_spike = np.append(latest_spike,max(C[I]))
        return latest_spike

    def Layer_2_connection_strengths_to_Layer_1(self):
        nL1 = 10
	nL2 = 15
        N = self.network.parameters.N
	Q1, Q2 = self.network.Q
	indxs = np.zeros((nL2, nL1))
        min_con_shown = np.inf*np.ones(nL2)
	for n in range(nL2):
	    v = Q2[:, n].copy()
	    for c in range(nL1):
        	idx = np.argmax(v)
                if min_con_shown[n] > v[idx]:
                    min_con_shown[n] = v[idx]
	        indxs[n, c] = idx
	        v[idx] = 0
	L2C = np.zeros((nL1*nL2, N))
	for ii, n in enumerate(indxs.ravel()):
            ii_2 = int(ii/nL1)
            rf = Q1[:, n]/(Q1[:, n]**2).sum()
            rf = rf-rf.min()
            rf = rf/rf.max()
	    L2C[ii] = np.power(np.log(abs(Q2[n, ii_2])/min_con_shown[ii_2]), .25)*rf
            print ii_2, np.log(abs(Q2[n, ii_2])/min_con_shown[ii_2])

	fig=plt.figure()
	side = int(np.sqrt(N))
	img = tile_raster_images(L2C, img_shape = (side,side),
                                 tile_shape = (nL2, nL1), tile_spacing=(4, 1),
                                 scale_rows_to_unit_interval=False,
                                 output_pixel_vals=False)
	plt.imshow(img,cmap=plt.cm.Greys, interpolation='nearest')
	#plt.title('Layer 2 Connection Strengths to Layer 1')
	plt.xlabel('Layer 1 Receptive Fields')
	plt.ylabel('Layer 2 Neurons')
        plt.xticks([])
        plt.yticks([])
        self.pp.savefig(fig)
        plt.close(fig)

	Y2 = self.network.Y[1]
	sort_idxs = np.argsort(Y2.sum(axis=0))[::-1][:nL2]
        min_con_shown = np.inf*np.ones(nL2)
        for n, idx in enumerate(sort_idxs):
            v = Q2[:, idx].copy()
            for c in range(nL1):
                idx = np.argmax(v)
                if min_con_shown[n] > v[idx]:
                    min_con_shown[n] = v[idx]
                indxs[n,c] = idx
                v[idx] = 0
        L2C=np.zeros((nL1*nL2, N))
        for ii, n in enumerate(indxs.ravel()):
            ii_2 = int(ii/nL1)
            rf = Q1[:, n]/(Q1[:, n]**2).sum()
            rf = rf-rf.min()
            rf = rf/rf.max()
            L2C[ii] = np.power(np.log(abs(Q2[n, sort_idxs[ii_2]])/min_con_shown[ii_2]), .25)*rf
            print ii_2, np.log(abs(Q2[n, sort_idxs[ii_2]])/min_con_shown[ii_2])

        fig=plt.figure()
        side = int(np.sqrt(N))
        img = tile_raster_images(L2C, img_shape = (side,side),
                                 tile_shape = (nL2, nL1), tile_spacing=(4, 1),
                                 scale_rows_to_unit_interval=False,
                                 output_pixel_vals=False)
        plt.imshow(img,cmap=plt.cm.Greys, interpolation='nearest')
        #plt.title('Sort Layer 2 Connection Strengths to Layer 1')
        plt.xlabel('Layer 1 Receptive Fields')
	plt.ylabel('Sorted Layer 2 Neurons')
        plt.xticks([])
        plt.yticks([])
        self.pp.savefig(fig)
        plt.close(fig)

        
    def PlotAll(self):
        self.validation_data()
        with PdfPages(self.directory+'/Images/plots.pdf') as self.pp:
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
                self.Plot_Raster(layer)
                self.Plot_Rate_vs_Time(layer)
                self.frame_spike_correlation()
                self.Plot_Rate_Hist_LC(layer)
            if self.network.n_layers > 1:
                self.Layer_2_connection_strengths_to_Layer_1()

if __name__ == "__main__":
    directory = sys.argv[1]
    plotter = Plot(directory)
    plotter.load_network()
    plotter.PlotAll()
