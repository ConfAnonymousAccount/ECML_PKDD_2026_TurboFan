import copy
from typing import Iterable, Union
import numpy as np

from package.data.scaler import TimeSeriesNormalization

class MinMaxScaler(TimeSeriesNormalization):
    """
    Min Max Scaler
    """
    def __init__(self, datasets: tuple):
        super().__init__(datasets)

        self.min_x = None
        self.max_x = None
        self.min_y = 0
        self.max_y = 100

    def fit(self, return_params: bool=False) -> Union[None, tuple]:
        """Fit the normalizer on training data to obtain the parameters

        Parameters
        ----------
        return_params, optional
            if to return fitted parameters, by default False

        Returns
        -------
            fitted parameters for normalization
        """
        train_x_obs = []
        train_y_obs = []

        for seq_x, seq_y, seq_len in zip(self.Xtrain, self.Ytrain, self.seq_len_train):
            train_x_obs.append(seq_x[:seq_len])
            train_y_obs.append(seq_y[:seq_len].reshape(-1,))

        train_x_obs = np.vstack(train_x_obs)
        train_y_obs = np.hstack(train_y_obs)

        self._aux_compute_params(train_x_obs, train_y_obs)

        if return_params:
            return (self.min_x, self.max_x), (self.min_y, self.max_y)

    def _aux_compute_params(self, x_train: np.ndarray, y_train: np.ndarray):
        """Helper function to compute parameters

        Parameters
        ----------
        x_train
            the training features in the shape [data_size x input_size]
        y_train
            the target variable in the shape [data_size x output_size]
        """
        self.min_x = np.min(x_train, axis=0)
        self.max_x = np.max(x_train, axis=0)
        self.min_y = np.min(y_train, axis=0)
        self.max_y = np.max(y_train, axis=0)

    def transform(self, datasets: tuple=("train", "val", "test")):
        """Apply the normalization on datasets

        Parameters
        ----------
        datasets, optional
            on which datasets normalization should be applied, by default ("train", "val", "test")
        """
        self.Xstrain = copy.deepcopy(self.Xtrain)
        self.Ystrain = copy.deepcopy(self.Ytrain)
        self.Xsval = copy.deepcopy(self.Xval)
        self.Ysval = copy.deepcopy(self.Yval)
        self.Xstest = copy.deepcopy(self.Xtest)
        self.Ystest = copy.deepcopy(self.Ytest)

        if "train" in datasets:
            for seq_x, seq_y, seq_len in zip(self.Xstrain, self.Ystrain, self.seq_len_train):
                seq_x[:seq_len] = (seq_x[:seq_len] - self.min_x) / (self.max_x - self.min_x)
                seq_y[:seq_len] = (seq_y[:seq_len] - self.min_y) / (self.max_y - self.min_y)

        if "val" in datasets:
            for seq_x, seq_y, seq_len in zip(self.Xsval, self.Ysval, self.seq_len_val):
                seq_x[:seq_len] = (seq_x[:seq_len] - self.min_x) / (self.max_x - self.min_x)
                seq_y[:seq_len] = (seq_y[:seq_len] - self.min_y) / (self.max_y - self.min_y)

        if "test" in datasets:
            for seq_x, seq_y, seq_len in zip(self.Xstest, self.Ystest, self.seq_len_test):
                seq_x[:seq_len] = (seq_x[:seq_len] - self.min_x) / (self.max_x - self.min_x)
                seq_y[:seq_len] = (seq_y[:seq_len] - self.min_y) / (self.max_y - self.min_y)

    def inverse_transform(self, pred):
        """inverse transform the normalization

        Parameters
        ----------
        pred
            the predictions obtained by a machine learning method

        Returns
        -------
            return denormalized data
        """
        return (pred * (self.max_y - self.min_y)) + self.min_y
