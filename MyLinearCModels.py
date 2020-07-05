import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .MyLinearModels import MyLinearModels


class MyLinearCModels(MyLinearModels):
    """Represents a Linear C model"""

    def __init__(self, sim_size=20, test_size=0.25):
        """
        Instantiates an instance of MyLinearCModels

        Parameters
        ----------
        sim_size : int, default=20
            Number of simulations
        test_size : float, default=0.25
            Test size for split
        """
        super().__init__(sim_size, test_size)
        self.setting = 'C'

    def get_weights_byparam(self, clist):
        """Returns weights by C in clist"""
        # Check if specified param list is valid
        if not set(clist).issubset(set(self.clist)):
            raise RuntimeError('MyLinearCModels: C in clist '
                               'are not trained')
        # Get indexes of parameters
        indexes = [self.clist.index(c) for c in clist]
        weights = self.list_weight
        if not len(weights):
            raise RuntimeError('MyLinearCModels: model not yet trained')
        # Get weights per parameter
        param_coefs = {self.clist[ind]: list() for ind in indexes}
        for i in range(len(self.simulations)):
            # Get weights per simulation
            start = i * len(self.clist)
            coefs = weights[start:start + len(self.clist)]
            for ind in indexes:
                # Get weight of each parameter per simulation
                param_coefs[self.clist[ind]].append(coefs[ind])
        # Convert values into np.array to convert to DataFrame
        param_coefs = {k: np.asarray(v) for k, v in param_coefs.items()}
        return {k: pd.DataFrame(v.reshape(-1, v.shape[-1])).mean(axis=0)
                for k, v in param_coefs.items()}

    def create_weightplot_axes(self, clist, ax):
        """
        Returns matplotlib.axes.Axes for weight for each clist.
        Parameter should have been trained before calling.

        Parameters
        ----------
        clist : list
            C of weights to plot
        ax : matplotlib.axes.Axes
            matplotlib.axes.Axes to fill

        Returns
        -------
        create_alphaplot_axes : matplotlib.axes.Axes
            matplotlib.axes.Axes for alpha plot
        """
        # Plot each alpha
        param_dict = self.get_weights_byparam(clist)
        return super()._create_weightplot_axes(param_dict, ax=ax)