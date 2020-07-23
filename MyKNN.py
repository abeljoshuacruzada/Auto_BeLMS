import numpy as np
import pandas as pd
import warnings
import time
from .MyMLModels import MyMLModels


class MyKNN(MyMLModels):
    """Represents a KNN model"""

    def __init__(self, sim_size=20, test_size=0.25):
        """
        Instantiates an instance of MyLogisticRegression

        Parameters
        ----------
        sim_size : int, default=20
            Number of simulations
        test_size : float, default=0.25
            Test size for split
        """
        super().__init__(sim_size, test_size)
        self.setting = 'n neighbors'
        self.leaf_size = 30
        self.p = 2
        self.metric = 'minkowski'

    def get_bestparameter(self):
        """Return the best parameter"""
        if self._df_test is None:
            raise RuntimeError('get_bestparameter: please the '
                                   'train model first')
        mean = self._df_test.mean(axis=1)
        if len(mean) == 1:
            result = mean.idxmax()
        elif len(mean) == 2:
            result = mean.loc[mean.index > 1].idxmax()
        else:
            result = mean.loc[mean.index > 2].idxmax()
        return result