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
from sklearn.neighbors import KNeighborsClassifier
from tqdm.autonotebook import tqdm
from .MyMLModels import MyMLModels


class MyKNNClassifier(MyMLModels):
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
        self.setting = 'n neighbors'

    def train_model(self, X, y, nlist=None, n_jobs=None):
        """
        Train KNN model
        Return 2 tuple of pandas DataFrame for train and test
        accuracy score

        Parameters
        ----------
        X : list or array type
            Features
        y : list or array type
            Target
        nlist : list of int, optional, default=None
            A list of n neighbors to be tested in the model.
        n_jobs : int, default=None
            The number of parallel jobs to run for neighbors search.
            None means 1 unless in a joblib.parallel_backend context.
            -1 means using all processors.
        """
        self._X = X
        self._y = y
        if nlist is None:
            nlist = self.nlist
        else:
            self.nlist = nlist
        # Result dataframes
        self._df_train = pd.DataFrame()
        self._df_test = pd.DataFrame()
        with tqdm(total=len(self.simulations)*len(nlist)) as pb:
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
                for n in nlist:
                    # Create KNN classifier for each n
                    cls = KNeighborsClassifier(n,
                                               n_jobs=-1
                                               ).fit(X_train, y_train)
                    # Store result of each n
                    accuracy_train.append(cls.score(X_train, y_train))
                    accuracy_test.append(cls.score(X_test, y_test))
                    pb.update(1)

                # Store result for each simulation
                self._df_train[seed] = accuracy_train
                self._df_test[seed] = accuracy_test
        self._df_train.index = nlist
        self._df_test.index = nlist
        return self._df_train, self._df_test

    def get_toppredictors(self, n_neighbors=None):
        """
        Return the top predictors, model must be first trained

        Parameters
        ----------
        n_neighbors : int, default=None
            Number of neighbors to use by default for
            :meth:`kneighbors` queries. If None get the best parameter
            after the model has been trained

        Returns
        -------
        get_toppredictors : pandas.DataFrame
            top predictors of model
        """
        if self._df_test is None:
            raise RuntimeError('MyKNNClassifier: please train the '
                               'model first')
        elif n_neighbors is None:
            n = self.get_bestparameter()
        else:
            n = n_neighbors

        knn = KNeighborsClassifier(n).fit(self._X, self._y)
        df_score = pd.DataFrame()
        features = self._X.columns
        for i in range(len(features)):
            x = self._X.iloc[:, i].to_numpy().reshape(-1, 1)
            df_score[i] = cross_val_score(knn, x, self._y)
        # Get score sorted by highest
        df_score = df_score.mean()
        df_score.index = features
        df_score = (df_score.sort_values(ascending=False)
                    .to_frame().rename(columns={0: 'score'}))
        return df_score

    def get_metric(self, X=None, y=None,
                   random_state=None, n_neighbors=None, ax=None):
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
        random_state : int, RandomState instance, default=None
            Used to shuffle the data.
        n_neighbors : int, default=None
            Number of neighbors to use by default for
            :meth:`kneighbors` queries. If None get the best parameter
            after the model has been trained
        ax : matplotlib.axes.Axes, default=None
            matplotlib.axes.Axes to fill confusion matrix

        Returns
        -------
        get_confusionmatrix : dictionary
            Confusion matrix, classification report
            {'confmat'=val, 'report'=val, 'ax':val}
        """
        param = self._set_data(X, y, random_state, n_neighbors)
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
            knn = (KNeighborsClassifier(param['parameter'])
                   .fit(X_train, y_train))
        y_pred = knn.predict(X_test)

        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        report = pd.DataFrame(classification_report(y_true=y_test,
                                                    y_pred=y_pred,
                                                    output_dict=True)).T

        if ax is not None:
            # Plot confusion matrix
            ax = self._fill_confusionmatrix(ax, confmat)
        return dict(confmat=confmat, report=report, ax=ax)
    
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