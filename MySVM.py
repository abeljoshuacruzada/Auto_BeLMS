import numpy as np
import pandas as pd
import warnings
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from tqdm.autonotebook import tqdm
from .MyMLModels import MyMLModels


class MySVM(MyMLModels):
    """Represents a Non-linear SVM model"""

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
        self.setting = 'gamma'
        self.log = True
        self.kernel = 'rbf'
        self.degree = 3
        self.C = 1.0
        self.coef0 = 0.0
        self.max_iter = 100
        self.random_state = None

    def train_model(self, X, y, kernel='rbf', degree=3, C=1.0,
                    coef0=0.0, random_state=None,
                    max_iter=100, glist=None):
        """
        Train Non-linear SVM model
        Return 2 tuple of pandas DataFrame for train and test
        accuracy score

        Parameters
        ----------
        X : pandas DataFrame
            Features
        y : pandas DataFrame
            Target
        kernel : {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’},
                    default=’rbf’
            Specifies the kernel type to be used in the algorithm.
            It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
            ‘precomputed’ or a callable. If none is given, ‘rbf’
            will be used. If a callable is given it is used to pre-compute
            the kernel matrix from data matrices; that matrix should
            be an array of shape
        degree : int, default=3
            Degree of the polynomial kernel function (‘poly’).
            Ignored by all other kernels.
        C : float, default=1.0
            Regularization parameter. The strength of the regularization is
            inversely proportional to C. Must be strictly positive.
            The penalty is a squared l2 penalty.
        coef0 : float, default=0.0
            Independent term in kernel function. It is only significant
            in ‘poly’ and ‘sigmoid’.
        random_state : int or RandomState instance, default=None
            Controls the pseudo random number generation for shuffling
            the data for probability estimates.
            Ignored when probability is False.
            Pass an int for reproducible output across
            multiple function calls.
        max_iter : int, default='100'
            Maximum number of iterations taken for the solvers to converge.
        glist : list of floats, optional, default=None
            A list of gamma to be tested in the model.
        """
        self._X = X
        self._y = y
        self.kernel = kernel
        self.degree = degree
        self.C = C
        self.coef0 = coef0
        self.max_iter = max_iter
        self.random_state = random_state

        if glist is None:
            glist = self.glist
        else:
            self.glist = glist
        # Result dataframes
        self._df_train = pd.DataFrame()
        self._df_test = pd.DataFrame()
        with tqdm(total=len(self.simulations)*len(glist)) as pb:
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
                for g in glist:
                    # Create Non linear SVM for each gamma
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore",
                                                category=ConvergenceWarning)
                        cls = SVC(kernel=kernel,
                                  degree=degree,
                                  gamma=g,
                                  coef0=coef0,
                                  C=C,
                                  max_iter=max_iter,
                                  random_state=random_state
                                  ).fit(X_train, y_train)
                    # Store result of each gamma
                    accuracy_train.append(cls.score(X_train, y_train))
                    accuracy_test.append(cls.score(X_test, y_test))
                    pb.update(1)

                # Store result for each simulation
                self._df_train[seed] = accuracy_train
                self._df_test[seed] = accuracy_test
        self._df_train.index = glist
        self._df_test.index = glist
        return self._df_train, self._df_test

    def get_toppredictors(self, gamma=None):
        """
        Return the top predictors, model must be first trained

        Parameters
        ----------
        gamma : float, default=None
            gamma to use. If None get the best parameter
            after the model has been trained

        Returns
        -------
        get_toppredictors : pandas.DataFrame
            top predictors of model
        """
        if self._df_test is None:
            raise RuntimeError('MySVM: please train the '
                               'model first')
        elif gamma is None:
            gamma = self.get_bestparameter()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    category=ConvergenceWarning)
            svc = SVC(kernel=self.kernel,
                      degree=self.degree,
                      gamma=gamma,
                      coef0=self.coef0,
                      C=self.C,
                      max_iter=self.max_iter,
                      random_state=self.random_state
                      ).fit(self._X, self._y)

            df_score = pd.DataFrame()
            features = self._X.columns
            for i in range(len(features)):
                x = self._X.iloc[:, i].to_numpy().reshape(-1, 1)
                df_score[i] = cross_val_score(svc, x, self._y)
        # Get score sorted by highest
        df_score = df_score.mean()
        df_score.index = features
        df_score = (df_score.sort_values(ascending=False)
                    .to_frame().rename(columns={0: 'score'}))
        return df_score

    def get_metric(self, X=None, y=None, kernel='rbf', degree=3, gamma=None,
                   C=1.0, coef0=0.0, max_iter=100,
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
        kernel : {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’},
                    default=’rbf’
            Specifies the kernel type to be used in the algorithm.
            It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
            ‘precomputed’ or a callable. If none is given, ‘rbf’
            will be used. If a callable is given it is used to pre-compute
            the kernel matrix from data matrices; that matrix should
            be an array of shape
        degree : int, default=3
            Degree of the polynomial kernel function (‘poly’).
            Ignored by all other kernels.
        gamma : float, default=None
            gamma to be tested in the model.
        C : float, default=1.0
            C parameter. If None get the best parameter
            after the model has been trained
        coef0 : float, default=0.0
            Independent term in kernel function. It is only significant
            in ‘poly’ and ‘sigmoid’.
        max_iter : int, default='100'
            Maximum number of iterations taken for the solvers to converge.
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
        param = self._set_data(X, y, random_state, gamma)
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
            svc = SVC(kernel=kernel,
                      degree=degree,
                      gamma=param['parameter'],
                      coef0=coef0,
                      C=C,
                      max_iter=max_iter,
                      random_state=random_state
                      ).fit(X_train, y_train)
        y_pred = svc.predict(X_test)

        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        report = pd.DataFrame(classification_report(y_true=y_test,
                                                    y_pred=y_pred,
                                                    output_dict=True)).T

        if ax is not None:
            # Plot confusion matrix
            ax = self._fill_confusionmatrix(ax, confmat)
        return dict(confmat=confmat, report=report, ax=ax)