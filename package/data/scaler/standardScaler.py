import copy
import numpy as np

from package.data.scaler import TimeSeriesNormalization

class StandardScaler(TimeSeriesNormalization):
    """
        Standard the date, that is scale the variable to a same measurement unit

        This function applies ``x - mean(x) / std(x)`` to time series data

        The mean and standard deviation are computed on training dataset and are applied on validation and test
        sets to scale them. 
    """
    def __init__(self, datasets: tuple):
        super().__init__(datasets)

        self.mean_X = None
        self.std_X = None
        self.mean_Y = None
        self.std_Y = None


    def fit(self, return_params=False):
        Xtrain_sequences = []
        Ytrain_sequences = []

        for i in range(len(self.Xtrain)):
            Xtrain_sequences.append(self.Xtrain[i][:self.seq_len_train[i]])
            Ytrain_sequences.append(self.Ytrain[i][:self.seq_len_train[i]])

        Xtrain_sequences = np.vstack(Xtrain_sequences)
        Ytrain_sequences = np.hstack(Ytrain_sequences)

        self._aux_compute_params(Xtrain_sequences, Ytrain_sequences)

        if return_params:
            return (self.mean_X, self.std_X), (self.mean_Y, self.std_Y)

    def _aux_compute_params(self, X, Y):
        self.mean_X = np.mean(X, axis=0)
        self.std_X = np.std(X, axis=0)
        self.mean_Y = np.mean(Y, axis=0)
        self.std_Y = np.std(Y, axis=0)

    def transform(self, datasets=("train", "val", "test")):
        if "train" in datasets:
            for i in range(len(self.Xtrain)):
                self.Xstrain[i][:self.seq_len_train[i]] = (self.Xtrain[i][:self.seq_len_train[i]] - self.mean_X) / self.std_X
                self.Ystrain[i][:self.seq_len_train[i]] = (self.Ytrain[i][:self.seq_len_train[i]] - self.mean_Y) / self.std_Y

        if "val" in datasets:
            for i in range(len(self.Xval)):
                self.Xsval[i][:self.seq_len_val[i]] = (self.Xval[i][:self.seq_len_val[i]] - self.mean_X) / self.std_X
                self.Ysval[i][:self.seq_len_val[i]] = (self.Yval[i][:self.seq_len_val[i]] - self.mean_Y) / self.std_Y

        if "test" in datasets:
            for i in range(len(self.Xtest)):
                self.Xstest[i][:self.seq_len_test[i]] = (self.Xtest[i][:self.seq_len_test[i]] - self.mean_X) / self.std_X
                self.Ystest[i][:self.seq_len_test[i]] = (self.Ytest[i][:self.seq_len_test[i]] - self.mean_Y) / self.std_Y

    def inverse_transform(self, pred):
        return (pred * self.std_Y) + self.mean_Y
