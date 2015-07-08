import theano
import numpy as np

class Monitor(object):
    def __init__(self, network, learn):
        self.network = network
        self.learn = learn
        self.parameters = network.parameters
        self.num_trials = self.parameters.num_trials
        self.SNR = np.zeros(self.num_trials)
        self.y_bar = np.zeros(self.num_trials)
        self.Cyy_bar = np.zeros(self.num_trials)
        self.mag_dW = np.zeros(self.num_trials)
        
        X = self.network.X
        Y = self.network.Y
        Q = self.network.Q
        SNR = X.std()**2/(X-Y.dot(Q.T)).std()**2
        y_bar = network.Y.mean()
        Cyy_bar = (network.Y.T.dot(network.Y)/network.parameters.batch_size).mean()
        self.f = theano.function([], [SNR, y_bar, Cyy_bar])


    def log(self,tt):
        SNR, y_bar, Cyy_bar = self.f()
        self.SNR[tt] = SNR
        self.y_bar[tt] = y_bar
        self.Cyy_bar[tt] = Cyy_bar
        self.mag_dW[tt] = self.learn.mag_dW[0]