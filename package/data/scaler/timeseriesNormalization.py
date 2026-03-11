import copy
import numpy as np


class TimeSeriesNormalization(object):
    """
    class dedicated to normalize the time series data with various approaches and methods
    """
    def __init__(self,
                 datasets: tuple
                ):
        super(TimeSeriesNormalization, self).__init__()

        self.datasets = datasets
        (self.Xtrain, self.Ytrain, self.seq_len_train), \
            (self.Xval, self.Yval, self.seq_len_val), \
            (self.Xtest, self.Ytest, self.seq_len_test) = self.datasets

        self.Xstrain = copy.deepcopy(self.Xtrain)
        self.Ystrain = copy.deepcopy(self.Ytrain)
        self.Xsval = copy.deepcopy(self.Xval)
        self.Ysval = copy.deepcopy(self.Yval)
        self.Xstest = copy.deepcopy(self.Xtest)
        self.Ystest = copy.deepcopy(self.Ytest)


    def fit(self, return_params=False):
        """fit the normalizer on data to estimate the parameters"""
        pass


    def transform(self, datasets=("train", "val", "test")):
        """
        Transform (scale) the data with respect to the computed paramaters
        """
        pass

    def inverse_transform(self, pred) -> tuple:
        """
        Inverse the tranform for outputs (Ys)
        """
        pass

    def get_datasets(self) -> tuple:
        """function to get the normalized datasets

        Returns
        -------
            a set of tuples representing normalized datasets for training, validation and test
        """
        return (self.Xstrain, self.Ystrain), (self.Xsval, self.Ysval), (self.Xstest, self.Ystest)
