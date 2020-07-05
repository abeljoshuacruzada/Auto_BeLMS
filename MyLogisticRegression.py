import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from tqdm.autonotebook import tqdm
from .MyLinearCModels import MyLinearCModels


class MyLogisticRegression(MyLinearCModels):
    """Represents a logistic regression model"""

    def train_model(self, X, y, penalty='l2', solver='lbfgs', max_iter=100,
                    dual=False, random_state=None, n_jobs=None, clist=None):
        """
        Train logistic regression model
        Return 2 tuple of pandas DataFrame for train and test
        accuracy score

        Parameters
        ----------
        X : pandas DataFrame
            Features
        y : pandas DataFrame
            Target
        penalty : {'l1', 'l2', 'elasticnet', 'none'}, optional, default='l2'
            Used to specify the norm used in the penalization.
        solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, optional
            Algorithm to use in the optimization problem. Default is 'lbfgs'.
        max_iter : int, default=100
            Maximum number of iterations taken for the solvers to converge.
        dual : bool, default=False
            Dual or primal formulation. Dual formulation is only
            implemented for l2 penalty with liblinear solver.
            Prefer dual=False when n_samples > n_features.
        random_state : int, RandomState instance, default=None
            Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to
            shuffle the data.
        n_jobs : int, default=None
            Number of CPU cores used when parallelizing over classes if
            multi_class='ovr'". This parameter is ignored when the ``solver``
            is set to 'liblinear' regardless of whether 'multi_class' is
            specified or not. ``None`` means 1 unless in
            a :obj:`joblib.parallel_backend`
            context. ``-1`` means using all processors.
            See :term:`Glossary <n_jobs>` for more details.
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
                    # Create logistic regression for each c
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore",
                                                category=ConvergenceWarning)
                        cls = LogisticRegression(penalty=penalty,
                                                 solver=solver,
                                                 C=c,
                                                 max_iter=max_iter,
                                                 random_state=random_state,
                                                 n_jobs=n_jobs
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

    def get_metric(self, X=None, y=None, penalty='l2', solver='lbfgs', C=None,
                   max_iter=100, dual=False,
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
        penalty : {'l1', 'l2', 'elasticnet', 'none'}, optional, default='l2'
            Used to specify the norm used in the penalization.
        solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, optional
            Algorithm to use in the optimization problem. Default is 'lbfgs'.
        C : float, default=None
            C parameter. If None get the best parameter
            after the model has been trained
        max_iter : int, default=100
            Maximum number of iterations taken for the solvers to converge.
        dual : bool, default=False
            Dual or primal formulation. Dual formulation is only
            implemented for l2 penalty with liblinear solver.
            Prefer dual=False when n_samples > n_features.
        random_state : int, RandomState instance, default=None
            Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to
            shuffle the data.
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
            log = LogisticRegression(penalty=penalty,
                                     solver=solver,
                                     C=param['parameter'],
                                     max_iter=max_iter,
                                     random_state=random_state
                                     ).fit(X_train, y_train)
        y_pred = log.predict(X_test)

        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        report = pd.DataFrame(classification_report(y_true=y_test,
                                                    y_pred=y_pred,
                                                    output_dict=True)).T

        if ax is not None:
            # Plot confusion matrix
            ax = self._fill_confusionmatrix(ax, confmat)
        return dict(confmat=confmat, report=report, ax=ax)
