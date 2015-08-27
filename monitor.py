import theano
import theano.tensor as T
import numpy as np

class Monitor(object):
    self.training_values = ['Mean Firing Count', 'Mean Correlation', 'SNR', 'Normalized SNR']
    self.training_mean_std = ['Dictionary Norm', 'Inhibitory Weights', 'Thresholds', 'Data Norm', 'Reconstruction Norm']
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
            y_bar = network.Y.mean()
            Cyy_bar = (network.Y.T.dot(network.Y)/network.parameters.batch_size).mean()
            outputs.append([y_bar, Cyy_bar])

            X_rec = Y.dot(Q.T)
            X_rec_norm = T.sqrt(T.sum(T.sqr(X_rec),axis =1,keepdims=True))
            X_norm = T.sqrt(T.sum(T.sqr(X),axis =1,keepdims=True))
            X_rec_bar = X_rec_norm.mean()
            X_rec_std = X_rec_norm.std()
            outputs.append([X_rec_bar, X_rec_std])

            X_bar = X_norm.mean()
            X_std = X_norm.std()
            outputs.append([X_bar, X_std])

            SNR_Norm = T.mean(T.var(X,axis=0))/T.mean(T.var(X-X_rec*X_norm/X_rec_norm,axis=0))
            SNR = T.mean(T.var(X,axis=0))/T.mean(T.var(X-X_rec_norm,axis=0))
            outputs.append([SNR, SNR_Norm])
            
            Q_norm = T.sqrt(T.sum(T.sqr(Q), axis=0))
            Q_bar = Q_norm.mean()
            Q_std = Q_norm.std()
            outputs.append([Q_bar, Q_std])

            W_bar = W.mean()
            W_std = W.std()
            outputs.append([W_bar, W_std])

            theta_bar = theta.mean()
            theta_std = theta.std()
            outputs.append([theta_bar, theta_std])

        self.f = theano.function([], outputs)


    def log(self,tt):
        SNR,SNR_Norm,y_bar,Cyy_bar,X_rec_bar,X_rec_std,Q_bar,Q_std,W_bar,W_std,theta_bar,theta_std,X_norm_bar = self.f()
        results = self.f()
        for layer in range(self.network.n_layers):
            self.channels['Mean Firing Rate'][layer, tt] = results.pop()
            self.channels['Mean Correlation'][layer, tt] = results.pop()
            self.channels['Reconstruction Norm'][layer, tt, 0] = results.pop()
            self.channels['Reconstruction Norm'][layer, tt, 1] = results.pop()
            self.channels['Data Norm'][layer, tt, 0] = results.pop()
            self.channels['Data Norm'][layer, tt, 1] = results.pop()
            self.channels['SNR'][layer, tt] = results.pop()
            self.channels['Normalize SNR'][layer, tt] = results.pop()
            self.channels['Dictionary Norm'][layer, tt, 0] = results.pop()
            self.channels['Dictionary Norm'][layer, tt, 1] = results.pop()
            self.channels['Inhibitory Weights'][layer, tt, 0] = results.pop()
            self.channels['Inhibitory Weights'][layer, tt, 1] = results.pop()
            self.channels['Thresholds'][layer, tt, 0] = results.pop()
            self.channels['Thresholds'][layer, tt, 1] = results.pop()
            self.tt = tt
    
    def cleanup(self):
        self.network = None
