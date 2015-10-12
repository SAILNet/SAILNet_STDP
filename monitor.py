import theano
import theano.tensor as T
import numpy as np

class Monitor(object):
    training_values = ['Mean Firing Count', 'Mean Covariance', 'SNR', 'Normalized SNR']
    training_mean_std = ['Dictionary Norm', 'Inhibitory Weights', 'Thresholds', 'Data Norm', 'Reconstruction Norm']
    def __init__(self, network):
        self.network = network
        self.parameters = network.parameters

        num_trails = self.parameters.num_trials
        n_layers = network.n_layers
        self.channels = {}

        for channel in self.training_values:
            self.channels[channel] = np.zeros((n_layers, num_trails))
        for channel in self.training_mean_std:
            self.channels[channel] = np.zeros((n_layers, num_trails, 2))

        outputs = []

        for layer in range(n_layers):
            if layer == 0:
                X = self.network.X
            else:
                X = self.network.Y[layer-1]
            Y = self.network.Y[layer]
            Q = self.network.Q[layer]
            W = self.network.W[layer]
            theta = self.network.theta[layer]
            y_bar = Y.mean()
            Cyy_bar = (Y.T.dot(Y)/network.parameters.batch_size).mean()
            outputs.extend([y_bar, Cyy_bar])

            X_rec = Y.dot(Q.T)
            X_rec_norm = T.sqrt(T.sum(T.sqr(X_rec),axis =1,keepdims=True))
            X_norm = T.sqrt(T.sum(T.sqr(X),axis =1,keepdims=True))
            X_rec_bar = X_rec_norm.mean()
            X_rec_std = X_rec_norm.std()
            outputs.extend([X_rec_bar, X_rec_std])

            X_bar = X_norm.mean()
            X_std = X_norm.std()
            outputs.extend([X_bar, X_std])

            SNR_Norm = T.mean(T.var(X,axis=0))/T.mean(T.var(X-X_rec*X_norm/X_rec_norm,axis=0))
            SNR = T.mean(T.var(X,axis=0))/T.mean(T.var(X-X_rec_norm,axis=0))
            outputs.extend([SNR, SNR_Norm])
            
            Q_norm = T.sqrt(T.sum(T.sqr(Q), axis=0))
            Q_bar = Q_norm.mean()
            Q_std = Q_norm.std()
            outputs.extend([Q_bar, Q_std])

            W_bar = W.mean()
            W_std = W.std()
            outputs.extend([W_bar, W_std])

            theta_bar = theta.mean()
            theta_std = theta.std()
            outputs.extend([theta_bar, theta_std])

            if time_data = True:
                Y_tm1 = self.network.Y_tm1[layer]
                firing_corr_t_tm1 = Y.dot(Y_tm1)
                outputs.extend([firing_corr_t_tm1])
            

        self.f = theano.function([], outputs)


    def log(self,tt):
        #SNR,SNR_Norm,y_bar,Cyy_bar,X_rec_bar,X_rec_std,Q_bar,Q_std,W_bar,W_std,theta_bar,theta_std,X_norm_bar = self.f()
        results = self.f()
        for layer in range(self.network.n_layers):
            self.channels['Mean Firing Count'][layer, tt] = results.pop(0)
            self.channels['Mean Covariance'][layer, tt] = results.pop(0)
            self.channels['Reconstruction Norm'][layer, tt, 0] = results.pop(0)
            self.channels['Reconstruction Norm'][layer, tt, 1] = results.pop(0)
            self.channels['Data Norm'][layer, tt, 0] = results.pop(0)
            self.channels['Data Norm'][layer, tt, 1] = results.pop(0)
            self.channels['SNR'][layer, tt] = results.pop(0)
            self.channels['Normalized SNR'][layer, tt] = results.pop(0)
            self.channels['Dictionary Norm'][layer, tt, 0] = results.pop(0)
            self.channels['Dictionary Norm'][layer, tt, 1] = results.pop(0)
            self.channels['Inhibitory Weights'][layer, tt, 0] = results.pop(0)
            self.channels['Inhibitory Weights'][layer, tt, 1] = results.pop(0)
            self.channels['Thresholds'][layer, tt, 0] = results.pop(0)
            self.channels['Thresholds'][layer, tt, 1] = results.pop(0)
            self.tt = tt
            self.firing_corr_t_tm1 = results.pop(0)

    def cleanup(self):
        self.network = None
