import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from .MyMLModels import MyMLModels


class MyLinearModels(MyMLModels):
    """Represents a linear ML model"""

    def __init__(self, sim_size=20, test_size=0.25):
        """
        Instantiates an instance of MyLinearModels

        Parameters
        ----------
        sim_size : int, default=20
            Number of simulations
        test_size : float, default=0.25
            Test size for split
        """
        super().__init__(sim_size, test_size)
        self.list_weight = []
        self.log = True

    def _get_weights(self):
        """Return the weights of features"""
        if not len(self.list_weight):
            raise RuntimeError('MyLinearModels: please train the model first')
        # Store weights to dataframe
        weight = np.asarray(self.list_weight)
        weight = weight.reshape(-1, weight.shape[-1])
        return pd.DataFrame(weight).mean()

    def get_toppredictors(self):
        """
        Return the top predictors based on weight, model
        must be first trained

        Returns
        -------
        get_toppredictors : pandas.DataFrame
            top predictors of model
        """
        weights = self._get_weights()
        df_weights = weights.to_frame()
        if type(self._X) == pd.DataFrame:
            df_weights.set_index(self._X.columns, inplace=True)
        df_weights.rename(columns={0: 'weight'}, inplace=True)
        return df_weights.apply(abs).sort_values(by='weight', ascending=False)
    
    def _create_weightplot_axes(self, param_dict, ax):
        """
        Returns matplotlib.axes.Axes for weight for each parameter.
        Parameter should have been trained before calling.

        Parameters
        ----------
        param_dict : dict
            dictionary where parameters are keys and values are
            features weights
        ax : matplotlib.axes.Axes
            matplotlib.axes.Axes to fill

        Returns
        -------
        create_alphaplot_axes : matplotlib.axes.Axes
            matplotlib.axes.Axes for alpha plot
        """
        # Plot each alpha
        for param, weights in param_dict.items():
            ax.plot(weights, 'o--', label=f'{param}')
        return ax