import theano
import numpy as np

class Monitor(object):
    def __init__(self, network):
        self.network = network
        self.num_trials = network.num_trials
        self.SNR = np.zeros(self.num_trials)
        self.y_bar = np.zeros(self.num_trials)
        self.Cyy_bar = np.zeros(self.num_trials)
        self.iter = 0

        X = self.network.X
        Y = self.network.Y
        Q = self.network.Q
        SNR = X.std()**2/(X-Y.dot(Q.T)).std()**2
        y_bar = network.Y.mean()
        Cyy_bar = (network.Y.T.dot(network.Y)/network.batch_size).mean()
        self.f = theano.function([], [SNR, y_bar, Cyy_bar])


    def log(self):
        SNR, y_bar, Cyy_bar = self.f()
        self.SNR[self.iter] = SNR
        self.y_bar[self.iter] = y_bar
        self.Cyy_bar[self.iter] = Cyy_bar
