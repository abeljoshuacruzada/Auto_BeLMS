import numpy as np
import pandas as pd
import warnings
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsRegressor
from tqdm.autonotebook import tqdm
from .MyKNN import MyKNN


class MyKNNRegressor(MyKNN):
    """Represents a KNN Regressor model"""

    def train_model(self, X, y, weights='uniform', leaf_size=30, p=2,
                    metric='minkowski', nlist=None, n_jobs=None):
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
        weights : {'uniform', 'distance'} or callable, default='uniform'
            weight function used in prediction.  Possible values:

            - 'uniform' : uniform weights.  All points in each neighborhood
              are weighted equally.
            - 'distance' : weight points by the inverse of their distance.
              in this case, closer neighbors of a query point will have a
              greater influence than neighbors which are further away.
            - [callable] : a user-defined function which accepts an
              array of distances, and returns an array of the same shape
              containing the weights.

            Uniform weights are used by default.
        leaf_size : int, default=30
            Leaf size passed to BallTree or KDTree.  This can affect the
            speed of the construction and query, as well as the memory
            required to store the tree.  The optimal value depends on the
            nature of the problem.
        p : int, default=2
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using manhattan_distance (l1),
            and euclidean_distance (l2) for p = 2. For arbitrary p,
            minkowski_distance (l_p) is used.
        metric : str or callable, default='minkowski'
            the distance metric to use for the tree.  The default metric is
            minkowski, and with p=2 is equivalent to the standard Euclidean
            metric. See the documentation of :class:`DistanceMetric` for a
            list of available metrics.
            If metric is "precomputed", X is assumed to be a distance matrix
            and must be square during fit. X may be a :term:`sparse graph`,
            in which case only "nonzero" elements may be considered neighbors.
        nlist : list of int, optional, default=None
            A list of n neighbors to be tested in the model.
        n_jobs : int, default=None
            The number of parallel jobs to run for neighbors search.
            None means 1 unless in a joblib.parallel_backend context.
            -1 means using all processors.
        """
        self._X = X
        self._y = y
        self.weights = weights
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
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
                    cls = KNeighborsRegressor(n,
                                              weights=weights,
                                              leaf_size=leaf_size,
                                              p=p,
                                              metric=metric,
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
            raise RuntimeError('MyKNNRegressor: please train the '
                               'model first')
        elif n_neighbors is None:
            n = self.get_bestparameter()
        else:
            n = n_neighbors

        knn = KNeighborsRegressor(n,
                                  weights=self.weights,
                                  leaf_size=self.leaf_size,
                                  p=self.p,
                                  metric=self.metric,
                                  n_jobs=-1
                                  ).fit(self._X, self._y)
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