import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC
from tqdm.autonotebook import tqdm
from .MyLinearCModels import MyLinearCModels


class MyLinearSVM(MyLinearCModels):
    """Represents a Linear SVM model"""

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
        self.setting = 'C'

    def train_model(self, X, y, penalty='l2', loss='squared_hinge',
                    max_iter=100, dual=False,
                    random_state=None, clist=None):
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
        penalty : {‘l1’, ‘l2’}, default=’l2’
            Specifies the norm used in the penalization. The ‘l2’
            penalty is the standard used in SVC. The ‘l1’ leads to
            coef_ vectors that are sparse.
        loss : {‘hinge’, ‘squared_hinge’}, default=’squared_hinge’
            Specifies the loss function. ‘hinge’ is the
            standard SVM loss (used e.g. by the SVC class)
            while ‘squared_hinge’ is the square of the hinge loss.
        max_iter : int, default='100'
            Maximum number of iterations taken for the solvers to converge.
        dual : bool, default=False
            Dual or primal formulation. Dual formulation is only
            implemented for l2 penalty with liblinear solver.
            Prefer dual=False when n_samples > n_features.
        random_state : int or RandomState instance, default=None
            Controls the pseudo random number generation for shuffling
            the data for probability estimates.
            Ignored when probability is False.
            Pass an int for reproducible output across
            multiple function calls.
        clist : list of floats, optional, default=None
            A list of C to be tested in the model.
        """
        self._X = X
        self._y = y
        if clist is None:
            clist = self.clist
        else:
            self.clist = clist
        # Result dataframes
        self._df_train = pd.DataFrame()
        self._df_test = pd.DataFrame()
        with tqdm(total=len(self.simulations)*len(clist)) as pb:
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
                for c in clist:
                    # Create linear SVM for each c
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore",
                                                category=ConvergenceWarning)
                        cls = LinearSVC(penalty=penalty,
                                        loss=loss,
                                        dual=dual,
                                        C=c,
                                        max_iter=max_iter,
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
        self._df_train.index = clist
        self._df_test.index = clist
        return self._df_train, self._df_test

    def get_metric(self, X=None, y=None, penalty='l2', loss='squared_hinge',
                   C=None, max_iter=100, dual=False,
                   random_state=None, ax=None):
        """
        Return a confusion matrix, classification report,
        and matplotlib.axes.Axes of confusion matrix
        in dictionary form

        Parameters
        ----------
        X : pandas DataFrame, default=None
            Features. If X=None or y=None will use the X of trained model
        y : pandas DataFrame, default=None
            Target. If X=None or y=None will use the y of trained model
        penalty : {‘l1’, ‘l2’}, default=’l2’
            Specifies the norm used in the penalization. The ‘l2’
            penalty is the standard used in SVC. The ‘l1’ leads to
            coef_ vectors that are sparse.
        loss : {‘hinge’, ‘squared_hinge’}, default=’squared_hinge’
            Specifies the loss function. ‘hinge’ is the
            standard SVM loss (used e.g. by the SVC class)
            while ‘squared_hinge’ is the square of the hinge loss.
        C : float, default=None
            C parameter. If None get the best parameter
            after the model has been trained
        max_iter : int, default='100'
            Maximum number of iterations taken for the solvers to converge.
        dual : bool, default=False
            Dual or primal formulation. Dual formulation is only
            implemented for l2 penalty with liblinear solver.
            Prefer dual=False when n_samples > n_features.
        random_state : int or RandomState instance, default=None
            Controls the pseudo random number generation for shuffling
            the data for probability estimates.
            Ignored when probability is False.
            Pass an int for reproducible output across
            multiple function calls.
        ax : matplotlib.axes.Axes, default=None
            matplotlib.axes.Axes to fill confusion matrix

        Returns
        -------
        get_confusionmatrix : dictionary
            Confusion matrix, classification report
            {'confmat'=val, 'report'=val, 'ax':val}
        """
        param = self._set_data(X, y, random_state, C)
        X = param['X']
        y = param['y']
        ds = train_test_split(X, y, test_size=self.test_size,
                              random_state=random_state)
        X_train = ds[0]
        X_test = ds[1]
        y_train = ds[2]
        y_test = ds[3]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    category=ConvergenceWarning)
            lsvm = LinearSVC(penalty=penalty,
                             loss=loss,
                             dual=dual,
                             C=param['parameter'],
                             max_iter=max_iter,
                             random_state=random_state
                             ).fit(X_train, y_train)
        y_pred = lsvm.predict(X_test)

        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        report = pd.DataFrame(classification_report(y_true=y_test,
                                                    y_pred=y_pred,
                                                    output_dict=True)).T

        if ax is not None:
            # Plot confusion matrix
            ax = self._fill_confusionmatrix(ax, confmat)
        return dict(confmat=confmat, report=report, ax=ax)
    
    def get_weights_byparam(self, clist):
        """Returns weights by C in clist"""
        # Check if specified param list is valid
        if not set(clist).issubset(set(self.clist)):
            raise RuntimeError('MyLogisticRegression: C in clist '
                               'are not trained')
        # Get indexes of parameters
        indexes = [self.clist.index(c) for c in clist]
        weights = self.list_weight
        if not len(weights):
            raise RuntimeError('MyLogisticRegression: model not yet trained')
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
        return {k: pd.DataFrame(v.reshape(-1, v.shape[1])).mean()
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