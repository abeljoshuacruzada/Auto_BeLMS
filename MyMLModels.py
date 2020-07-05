import numpy as np
import pandas as pd
import warnings
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class MyMLModels:
    """Represents machine learning models"""

    def __init__(self, sim_size=20, test_size=0.25):
        """
        Instantiates an instance of MyMLModels

        Parameters
        ----------
        sim_size : int, default=20
            Number of simulations
        test_size : float, default=0.25
            Test size for split
        """
        self.simulations = list(range(sim_size))
        self.test_size = test_size
        self.clist = [1e-8, 1e-4, 1e-3, .01, 0.1, 0.2,
                      0.4, 0.75, 1, 1.5, 3, 5, 10, 15, 20,
                      100, 1000]
        self.nlist = [1, 2, 3, 4, 5]
        self.glist = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.4, 0.75,
                      1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]
        self.alist = [0, 1e-12, 1e-10, 1e-8, 1e-4, 1e-3,0.1, 0.2,0.4,
                      0.75, 1, 1.5, 3, 5, 10, 15,  20]
        self._df_train = pd.DataFrame()
        self._df_test = pd.DataFrame()
        self._X = None
        self._y = None
        self.setting = ''
        self.log=False
        
    def get_bestparameter(self):
        """Return the best parameter"""
        if self._df_test is None:
            raise RuntimeError('get_bestparameter: please the '
                                   'train model first')
        return self._df_test.mean(axis=1).idxmax()

    def get_toppredictors(self):
        """Return the top predictors"""
        raise NotImplementedError     

    def create_errorband_axes(self, ax, df_train=None, df_test=None):
        """
        Returns matplotlib.axes.Axes for train and test dataset
        accuracy score errorband plots.

        Parameters
        ----------
        df_train : pandas.DataFrame, default=None
            Train dataset, if None will get the train dataset after
            train_model method has been called
        df_test : pandas.DataFrame, default=None
            Test dataset, if None will get the test dataset after
            train_model method has been called
        ax : matplotlib.axes.Axes
            matplotlib.axes.Axes to fill

        Returns
        -------
        create_errorband_axes : matplotlib.axes.Axes
            matplotlib.axes.Axes for train and test dataset
            accuracy score errorband plots.
        """
        if df_train is None or df_test is None:
            if self._df_train is None or self._df_test is None:
                raise RuntimeError('create_errorband_axes: please the '
                                   'train model first')
            else:
                df_train = self._df_train
                df_test = self._df_test
        dct = self.__get_bounds_traintest(df_train, df_test)
        xrange = df_train.index
        ax.plot(xrange, dct['mean_train'], label='train', color='tab:blue',
                marker='o')
        ax.plot(xrange, dct['mean_test'], label='test', color='tab:orange',
                marker='o')
        ax.fill_between(xrange, dct['mean_train'], dct['up_train'],
                        color='tab:blue',
                        alpha=0.5)
        ax.fill_between(xrange, dct['mean_train'], dct['low_train'],
                        color='tab:blue',
                        alpha=0.5)
        ax.fill_between(xrange, dct['mean_test'], dct['up_test'],
                        color='tab:orange',
                        alpha=0.5)
        ax.fill_between(xrange, dct['mean_test'], dct['low_test'],
                        color='tab:orange',
                        alpha=0.5)
        return ax

    def __get_bounds_traintest(self, df_train, df_test):
        """
        Return dictionary of the upper and lower bounds
        of train and test dataset

        Parameters
        ----------
        df_train : pandas.DataFrame
            Train dataset
        df_test : pandas.DataFrame
            Test dataset

        Returns
        -------
        __get_bounds_traintest : dictionary
            upper and lower bounds
            of train and test dataset
        """
        mean_train = df_train.mean(axis=1)
        mean_test = df_test.mean(axis=1)
        up_train = mean_train + df_train.std(axis=1)
        low_train = mean_train - df_train.std(axis=1)
        up_test = mean_test + df_train.std(axis=1)
        low_test = mean_test - df_train.std(axis=1)

        return dict(mean_train=mean_train,
                    up_train=up_train,
                    low_train=low_train,
                    mean_test=mean_test,
                    up_test=up_test,
                    low_test=low_test)

    def _set_data(self, X=None, y=None,
                  random_state=None,parameter=None):
        """
        Return X and y dataset, and parameter to be used
        in a form of dictionary

        Parameters
        ----------
        X : pandas DataFrame, default=None
            Features. If X=None or y=None will use the X of trained model
        y : pandas DataFrame, default=None
            Target. If X=None or y=None will use the y of trained model
        random_state : int, RandomState instance, default=None
            Used to shuffle the data.
        parameter : float or int, default=None
            Parameter to train the model.
            If None get the best parameter after the model has been trained

        Returns
        -------
        _set_data : dictionary
            X and y dataset, and parameter to be used
            {'X':val, 'y':val, 'parameter':val}
        """
        if X is None and y is None and self._df_test is None:
                raise RuntimeError('MyMLModels: please train the '
                                   'model first')
        if X is None or y is None:
            X = self._X
            y = self._y
        if parameter is None:
            parameter = self.get_bestparameter()
            
        return dict(X=X, y=y, parameter=parameter)
    
    def _fill_confusionmatrix(self, ax, confmat):
        """Fill matplotlib.axes.Axes with confusion matrix"""
        nrows = confmat.shape[0]
        ncols = confmat.shape[1]
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=3)
        for i in np.arange(nrows):
            for j in np.arange(ncols):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
        ax.set_xlabel('predicted label')
        ax.set_ylabel('true label')
        return ax