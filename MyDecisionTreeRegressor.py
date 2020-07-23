import numpy as np
import pandas as pd
import warnings
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeRegressor
from tqdm.autonotebook import tqdm
from .MyDecisionTree import MyDecisionTree


class MyDecisionTreeRegressor(MyDecisionTree):
    """Represents a Decision Tree Regressor model"""

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
        self.n_classes = None
        self.criterion = 'mse'

    def train_model(self, X, y, max_features=None, criterion='mse',
                    splitter='best', min_samples_split=2, min_samples_leaf=1,
                    random_state=None, depthlist=None):
        """
        Train Decision Tree Regressor model
        Return 2 tuple of pandas DataFrame for train and test
        accuracy score

        Parameters
        ----------
        X : pandas DataFrame
            Features
        y : pandas DataFrame
            Target
        max_features : int, float or {"auto", "sqrt", "log2"}, default=None
            The number of features to consider when looking for the best split:

                - If int, then consider `max_features` features at each split.
                - If float, then `max_features` is a fraction and
                  `int(max_features * n_features)` features are
                  considered at each split.
                - If "auto", then `max_features=sqrt(n_features)`.
                - If "sqrt", then `max_features=sqrt(n_features)`.
                - If "log2", then `max_features=log2(n_features)`.
                - If None, then `max_features=n_features`.

            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires
            to effectively inspect more than ``max_features`` features.
        criterion : {"mse", "friedman_mse", "mae"}, default="mse"
            The function to measure the quality of a split. Supported criteria
            are "mse" for the mean squared error, which is equal to variance
            reduction as feature selection criterion and minimizes the L2 loss
            using the mean of each terminal node, "friedman_mse", which uses mean
            squared error with Friedman's improvement score for potential splits,
            and "mae" for the mean absolute error, which minimizes the L1 loss
            using the median of each terminal node.

            .. versionadded:: 0.18
               Mean Absolute Error (MAE) criterion.
        splitter : {"best", "random"}, default="best"
            The strategy used to choose the split at each node. Supported
            strategies are "best" to choose the best split and "random" to
            choose
            the best random split.
        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:

            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
              `ceil(min_samples_split * n_samples)` are the minimum
              number of samples for each split.

            .. versionchanged:: 0.18
               Added float values for fractions.
        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left
            and right branches.  This may have the effect of smoothing the
            model, especially in regression.

            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
              `ceil(min_samples_leaf * n_samples)` are the minimum
              number of samples for each node.

            .. versionchanged:: 0.18
               Added float values for fractions.
        random_state : int, RandomState instance, default=None
            Controls the randomness of the estimator. The features are always
            randomly permuted at each split, even if ``splitter`` is set to
            ``"best"``. When ``max_features < n_features``, the algorithm will
            select ``max_features`` at random at each split before
            finding the best split among them. But the best found
            split may vary across different
            runs, even if ``max_features=n_features``. That is the case,
            if the improvement of the criterion is identical for
            several splits and one split has to be selected at random.
            To obtain a deterministic behaviour during fitting,
            ``random_state`` has to be fixed to an integer.
            See :term:`Glossary <random_state>` for details.
        depthlist : list int, default=None
            List of depth of tree to simulate.
        """
        self._X = X
        self._y = y
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.splitter = splitter
        self.random_state = random_state

        if depthlist is None:
            depthlist = self.depthlist
        else:
            self.depthlist = depthlist
        # Result dataframes
        self._df_train = pd.DataFrame()
        self._df_test = pd.DataFrame()
        with tqdm(total=len(self.simulations)*len(depthlist)) as pb:
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
                for d in depthlist:
                    # Create Decision Tree Regressor for each max depth
                    cls = DecisionTreeRegressor(
                        criterion=criterion,
                        splitter=splitter,
                        max_depth=d,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        random_state=random_state
                    ).fit(X_train, y_train)
                    # Store result of each gamma
                    accuracy_train.append(cls.score(X_train, y_train))
                    accuracy_test.append(cls.score(X_test, y_test))
                    pb.update(1)

                # Store result for each simulation
                self._df_train[seed] = accuracy_train
                self._df_test[seed] = accuracy_test
        self._df_train.index = depthlist
        self._df_test.index = depthlist
        return self._df_train, self._df_test

    def get_toppredictors(self, max_depth=None):
        """
        Return the top predictors, model must be first trained

        Parameters
        ----------
        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded
            until all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        Returns
        -------
        get_toppredictors : pandas.DataFrame
            top predictors of model
        """
        if self._df_test is None:
            raise RuntimeError('MyDecisionTreeRegressor: please train the '
                               'model first')
        elif max_depth is None:
            max_depth = self.get_bestparameter()

        tree = DecisionTreeRegressor(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state
        ).fit(self._X, self._y)

        df_score = pd.DataFrame()
        features = self._X.columns
        for i in range(len(features)):
            srs = pd.Series(tree.feature_importances_[i])
            df_score[i] = pd.Series(tree.feature_importances_[i])
        # Get score sorted by highest
        df_score = df_score.T
        df_score.index = features
        df_score = (df_score.sort_values(by=0, ascending=False)
                    .rename(columns={0: 'score'}))
        return df_score

    def get_metric(self, X=None, y=None, max_features=None, max_depth=None,
                   criterion='mse', splitter='best',
                   min_samples_split=2, min_samples_leaf=1,
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
        max_features : int, float or {"auto", "sqrt", "log2"}, default=None
            The number of features to consider when looking for the best split:

                - If int, then consider `max_features` features at each split.
                - If float, then `max_features` is a fraction and
                  `int(max_features * n_features)` features are
                  considered at each split.
                - If "auto", then `max_features=sqrt(n_features)`.
                - If "sqrt", then `max_features=sqrt(n_features)`.
                - If "log2", then `max_features=log2(n_features)`.
                - If None, then `max_features=n_features`.

            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires
            to effectively inspect more than ``max_features`` features.
        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded
            until all leaves are pure or until all leaves contain less than
            min_samples_split samples.
        criterion : {"mse", "friedman_mse", "mae"}, default="mse"
            The function to measure the quality of a split. Supported criteria
            are "mse" for the mean squared error, which is equal to variance
            reduction as feature selection criterion and minimizes the L2 loss
            using the mean of each terminal node, "friedman_mse", which uses mean
            squared error with Friedman's improvement score for potential splits,
            and "mae" for the mean absolute error, which minimizes the L1 loss
            using the median of each terminal node.

            .. versionadded:: 0.18
               Mean Absolute Error (MAE) criterion.
        splitter : {"best", "random"}, default="best"
            The strategy used to choose the split at each node. Supported
            strategies are "best" to choose the best split and "random" to
            choose
            the best random split.
        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:

            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
              `ceil(min_samples_split * n_samples)` are the minimum
              number of samples for each split.

            .. versionchanged:: 0.18
               Added float values for fractions.
        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left
            and right branches.  This may have the effect of smoothing the
            model, especially in regression.

            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
              `ceil(min_samples_leaf * n_samples)` are the minimum
              number of samples for each node.

            .. versionchanged:: 0.18
               Added float values for fractions.
        random_state : int, RandomState instance, default=None
            Controls the randomness of the estimator. The features are always
            randomly permuted at each split, even if ``splitter`` is set to
            ``"best"``. When ``max_features < n_features``, the algorithm will
            select ``max_features`` at random at each split before
            finding the best split among them. But the best found
            split may vary across different
            runs, even if ``max_features=n_features``. That is the case,
            if the improvement of the criterion is identical for
            several splits and one split has to be selected at random.
            To obtain a deterministic behaviour during fitting,
            ``random_state`` has to be fixed to an integer.
            See :term:`Glossary <random_state>` for details.
        ax : matplotlib.axes.Axes, default=None
            matplotlib.axes.Axes to fill confusion matrix

        Returns
        -------
        get_confusionmatrix : dictionary
            Confusion matrix, classification report
            {'confmat'=val, 'report'=val, 'ax':val}
        """
        param = self._set_data(X, y, random_state, max_depth)
        X = param['X']
        y = param['y']
        ds = train_test_split(X, y, test_size=self.test_size,
                              random_state=random_state)
        X_train = ds[0]
        X_test = ds[1]
        y_train = ds[2]
        y_test = ds[3]
        tree = DecisionTreeRegressor(criterion=criterion,
                                      splitter=splitter,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      max_features=max_features,
                                      random_state=random_state
                                      ).fit(X_train, y_train)
        y_pred = tree.predict(X_test)

        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        report = pd.DataFrame(classification_report(y_true=y_test,
                                                    y_pred=y_pred,
                                                    output_dict=True)).T

        if ax is not None:
            # Plot confusion matrix
            ax = self._fill_confusionmatrix(ax, confmat)
        return dict(confmat=confmat, report=report, ax=ax)