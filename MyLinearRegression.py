import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from tqdm.autonotebook import tqdm
from .MyLinearModels import MyLinearModels


class MyLinearRegression(MyLinearModels):
    """Represents a Linear regression model"""

    def __init__(self, sim_size=20, test_size=0.25):
        """
        Instantiates an instance of MyLinearRegression

        Parameters
        ----------
        sim_size : int, default=20
            Number of simulations
        test_size : float, default=0.25
            Test size for split
        """
        super().__init__(sim_size, test_size)
        self.setting = 'alpha'

    def train_model(self, X, y, regularizer='l2', fit_intercept=True,
                    normalize=False, max_iter=100, tol=0.001,
                    random_state=None, alist=None):
        """
        Train linear SVM model
        Return 2 tuple of pandas DataFrame for train and test
        accuracy score

        Parameters
        ----------
        X : pandas DataFrame
            Features
        y : pandas DataFrame
            Target
        regularizer : {‘l1’, ‘l2’}, default=l2
            Specifies the regularization to be used.
        fit_intercept : bool, default=True
            Whether to fit the intercept for this model. If set
            to false, no intercept will be used in calculations
            (i.e. ``X`` and ``y`` are expected to be centered).
        normalize : bool, default=False
            This parameter is ignored when ``fit_intercept`` is set to False.
            If True, the regressors X will be normalized before regression by
            subtracting the mean and dividing by the l2-norm.
            If you wish to standardize, please use
            :class:`sklearn.preprocessing.StandardScaler` before calling
            ``fit`` on an estimator with ``normalize=False``.
        max_iter : int, default='100'
            Maximum number of iterations taken for the solvers to converge.
        tol : float, default=1e-3
            Precision of the solution. (For lasso and ridge)
        random_state : int, RandomState instance, default=None
            Used when ``solver`` == 'sag' or 'saga' to shuffle the data.
            See :term:`Glossary <random_state>` for details.
            .. versionadded:: 0.17
           `random_state` to support Stochastic Average Gradient.
           (For lasso and ridge)
        alist : list of floats, optional, default=None
            A list of alpha to be tested in the model.  (For lasso and ridge)
        """
        self._X = X
        self._y = y
        if alist is None:
            alist = self.alist
        else:
            self.alist=alist
        # Result dataframes
        self._df_train = pd.DataFrame()
        self._df_test = pd.DataFrame()
        # Determine model to use
        if regularizer == 'l2':
            model = Ridge
        else:
            model = Lasso
        with tqdm(total=len(self.simulations)*len(alist)) as pb:
            for seed in self.simulations:
                ds = train_test_split(X, y, test_size=self.test_size,
                                      random_state=seed)
                pb.set_description(f'Iter: {seed + 1}')
                X_train = ds[0]
                X_test = ds[1]
                y_train = ds[2]
                y_test = ds[3]
                accuracy_train = []
                accuracy_test = []
                for a in alist:
                    # Create linear regression for each alpha
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        cls = model(alpha=a,
                                    fit_intercept=fit_intercept,
                                    normalize=normalize,
                                    max_iter=max_iter,
                                    tol=tol,
                                    random_state=random_state
                                    ).fit(X_train, y_train)
                    # Store result of each c
                    accuracy_train.append(cls.score(X_train, y_train))
                    accuracy_test.append(cls.score(X_test, y_test))
                    self.list_weight.append(cls.coef_)
                    pb.update(1)

                # Store result for each simulation
                self._df_train[seed] = accuracy_train
                self._df_test[seed] = accuracy_test
        self._df_train.index = alist
        self._df_test.index = alist
        return self._df_train, self._df_test
    
    def get_weights_byparam(self, alist):
        """Returns weights by alpha in alist"""
        # Check if specified param list is valid
        if not set(alist).issubset(set(self.alist)):
            raise RuntimeError('MyLinearRegression: alpha in alist '
                               'are not trained')
        # Get indexes of parameters
        indexes = [self.alist.index(a) for a in alist]
        weights = self.list_weight
        if not len(weights):
            raise RuntimeError('MyLinearRegression: model not yet trained')
        # Get weights per parameter
        param_coefs = {self.alist[ind]: list() for ind in indexes}
        for i in range(len(self.simulations)):
            # Get weights per simulation
            start = i * len(self.alist)
            coefs = weights[start:start + len(self.alist)]
            for ind in indexes:
                # Get weight of each parameter per simulation
                param_coefs[self.alist[ind]].append(coefs[ind])
        # Convert values into np.array to convert to DataFrame
        param_coefs = {k: np.asarray(v) for k, v in param_coefs.items()}
        return {k: pd.DataFrame(v.reshape(-1, v.shape[-1])).mean(axis=0)
                for k, v in param_coefs.items()}

    def create_weightplot_axes(self, alist, ax):
        """
        Returns matplotlib.axes.Axes for weight for each alist.
        Parameter should have been trained before calling.

        Parameters
        ----------
        alist : list
            alphas of weights to plot
        ax : matplotlib.axes.Axes
            matplotlib.axes.Axes to fill

        Returns
        -------
        create_alphaplot_axes : matplotlib.axes.Axes
            matplotlib.axes.Axes for alpha plot
        """
        # Plot each alpha
        param_dict = self.get_weights_byparam(alist)
        return super()._create_weightplot_axes(param_dict, ax=ax)