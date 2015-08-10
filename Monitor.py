import theano
import theano.tensor as T
import numpy as np

class Monitor(object):
    def __init__(self, network, learn):
        self.network = network
        self.learn = learn
        self.parameters = network.parameters
        self.num_trials = self.parameters.num_trials
        self.SNR = np.zeros(self.num_trials)
        self.SNR_Norm = np.zeros(self.num_trials)
        self.y_bar = np.zeros(self.num_trials)
        self.Cyy_bar = np.zeros(self.num_trials)
        self.mag_dW = np.zeros(self.num_trials)
	self.X_bar = np.zeros(self.num_trials)
	self.X_norm_bar = np.zeros(self.num_trials)
        self.X_rec_stats = np.zeros((self.num_trials,2))
        self.Q_stats = np.zeros((self.num_trials,2))
        self.W_stats = np.zeros((self.num_trials,2))
        self.theta_stats = np.zeros((self.num_trials,2))

        X = self.network.X
        Y = self.network.Y
        Q = self.network.Q
        W = self.network.W
        theta = self.network.theta
        
        X_rec = Y.dot(Q.T)
        X_rec_norm = T.sqrt(T.sum(T.sqr(X_rec),axis =1,keepdims=True))
        X_norm = T.sqrt(T.sum(T.sqr(X),axis =1,keepdims=True))

        SNR_Norm = T.mean(T.var(X,axis=0))/T.mean(T.var(X-X_rec*X_norm/X_rec_norm,axis=0))
        SNR = T.mean(T.var(X,axis=0))/T.mean(T.var(X-X_rec_norm,axis=0))
        
        #Q_norm = T.sqrt(T.sum(T.sqr(Q),axis =0))

        y_bar = network.Y.mean()
        Cyy_bar = (network.Y.T.dot(network.Y)/network.parameters.batch_size).mean()
	X_norm_bar = X.mean()
	X_rec_bar = X_rec.mean()
        X_rec_std = X_rec.std()
        Q_bar = Q.mean()
        Q_std = Q.std()
        W_bar = W.mean()
        W_std = W.std()
        theta_bar = theta.mean()
        theta_std = theta.std()

        self.f = theano.function([], [SNR,SNR_Norm,y_bar,Cyy_bar,X_rec_bar,X_rec_std,Q_bar,Q_std,W_bar,W_std,theta_bar,theta_std,X_norm_bar])


    def log(self,tt):
        SNR,SNR_Norm,y_bar,Cyy_bar,X_rec_bar,X_rec_std,Q_bar,Q_std,W_bar,W_std,theta_bar,theta_std,X_norm_bar = self.f()
        self.SNR[tt] = SNR
        self.SNR_Norm[tt] = SNR_Norm
        self.y_bar[tt] = y_bar
        self.Cyy_bar[tt] = Cyy_bar
	self.X_norm_bar[tt] = X_norm_bar
        self.X_rec_stats[tt,0] = X_rec_bar
        self.X_rec_stats[tt,1] = X_rec_std
        self.Q_stats[tt,0] = Q_bar
        self.Q_stats[tt,1] = Q_std
        self.W_stats[tt,0] = W_bar
        self.W_stats[tt,1] = W_std
        self.theta_stats[tt,0] = theta_bar
        self.theta_stats[tt,1] = theta_std        
        self.mag_dW[tt] = self.learn.mag_dW[0]
    
    def cleanup(self):
        self.network = None
        self.learn = None
