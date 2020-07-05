import numpy as np
import pandas as pd
import warnings
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.exceptions import ConvergenceWarning
from tqdm.autonotebook import tqdm
from .MyKNNClassifier import MyKNNClassifier
from .MyLogisticRegression import MyLogisticRegression
from .MyLinearSVM import MyLinearSVM
from .MySVM import MySVM
from .MyLinearRegression import MyLinearRegression


"""Models to simulate"""
classifier_models = {
    'KNN Classifier': {
        'model': MyKNNClassifier,
        'nlist': None,  # List of neighbors to simulate
        'n_jobs': None
    },
    'Logistic Regression L1': {
        'model': MyLogisticRegression,
        'penalty': 'l1',
        'solver': 'liblinear',
        'max_iter': 1000,
        'dual': False,
        'random_state': 0,
        'n_jobs': None,
        'clist': None  # If None, will simulate clist in MyMLModels
    },
    'Logistic Regression L2': {
        'model': MyLogisticRegression,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'dual': False,
        'random_state': 0,
        'n_jobs': None,
        'clist': None  # If None, will simulate clist in MyMLModels
    },
    'Linear SVM L1': {
        'model': MyLinearSVM,
        'penalty': 'l1',
        'loss': 'squared_hinge',
        'max_iter': 1000,
        'dual': False,
        'random_state': 0,
        'clist': None  # If None, will simulate clist in MyMLModels
    },
    'Linear SVM L2': {
        'model': MyLinearSVM,
        'penalty': 'l2',
        'loss': 'squared_hinge',
        'max_iter': 1000,
        'dual': True,
        'random_state': 0,
        'clist': None  # If None, will simulate clist in MyMLModels
    },
    'Non-Linear SVM RBF': {
        'model': MySVM,
        'kernel': 'rbf',
        'degree': 3,
        'C': 1.0,
        'coef0': 0.0,
        'random_state': 0,
        'max_iter': 1000,
        'glist': None  # If None, will simulate glist in MyMLModels
    },
}


regression_models = {
    'Lasso Regression L1': {
        'model': MyLinearRegression,
        'regularizer': 'l1',
        'fit_intercept': True,
        'normalize': False,
        'max_iter': 1000,
        'tol': 0.001,
        'random_state': None,
        'alist': None  # If None, will simulate alist in MyMLModels
    },
    'Ridge Regression L2': {
        'model': MyLinearRegression,
        'regularizer': 'l2',
        'fit_intercept': True,
        'normalize': False,
        'max_iter': 1000,
        'tol': 0.001,
        'random_state': None,
        'alist': None  # If None, will simulate alist in MyMLModels
    }
}


def simulate_classifiers(X, y, models, sim_size=20, test_size=0.25,
                         figsize=None, confmat=False):
    """
    Train all classifiermodels and returns pandas DataFrame of results
    models and return summary results of simulated models

    Parameters
    ----------
    X : pandas DataFrame
        Features
    y : pandas DataFrame
        Target
    models : dictionary
        models to simulate, must follow the structure of AutoBelms.model
        structure. Either remove or add new models
    sim_size : int, default=20
        simulation size
    test_size : float, default=0.25
        test dataset size
    confmat : bool, default=False
        display confusion matrix
    Returns
    -------
    simulate_classifiers : pandas DataFrame
        summary results of simulated models
    """
    data = []
    for mod_name, mod in models.items():
        mod_result = []
        print('Training {}'.format(mod_name))
        # Train the model
        start_time = time.time()
        model = mod['model'](sim_size=sim_size, test_size=test_size)
        df_train, df_test = model.train_model(X, y, *list(mod.values())[1:])
        train_time = time.time() - start_time
        print("%s seconds" % train_time)
        if figsize is not None:
            # Plot simulation
            plt.figure(figsize=figsize)
            ax = plt.gca()
            ax = model.create_errorband_axes(ax)
            ax.set_title(mod_name)
            ax.set_ylabel('Accuracy')
            ax.set_xlabel(model.setting)
            if model.log:
                ax.set_xscale('log')
            plt.legend()
            plt.show()
        if confmat:
            # Plot confusion matrix
            ax = plt.gca()
            metric = model.get_metric(ax=ax, random_state=0)
            ax.set_title('{} Confusion Matrix'.format(mod_name))
            plt.tight_layout()
            plt.show()
        else:
            metric = model.get_metric(ax=None, random_state=0)

        # Get results
        best_param = model.get_bestparameter()
        accuracy = df_test.mean(axis=1).loc[best_param]
        best_param = f'{model.setting} = {best_param}'
        try:
            top_predictor = model.get_toppredictors().idxmax()[0]
        except NotImplementedError:
            top_predictor = 'NA'
        # Store model results in a list
        mod_result.append(mod_name)
        mod_result.append(best_param)
        mod_result.append(accuracy)
        mod_result.append(top_predictor)
        mod_result.append(train_time)
        # Collect all models results
        data.append(mod_result)
    return pd.DataFrame(data, columns=['Model', 'Best Parameter',
                                       'Accuracy', 'Top Predictor',
                                       'Train Time'])


def simulate_regressor(X, y, models, sim_size=20, test_size=0.25,
                         figsize=None):
    """
    Train all regression models and returns pandas DataFrame of results
    models and return summary results of simulated models

    Parameters
    ----------
    X : pandas DataFrame
        Features
    y : pandas DataFrame
        Target
    models : dictionary
        models to simulate, must follow the structure of AutoBelms.model
        structure. Either remove or add new models
    sim_size : int, default=20
        simulation size
    test_size : float, default=0.25
        test dataset size
    Returns
    -------
    simulate_classifiers : pandas DataFrame
        summary results of simulated models
    """
    data = []
    for mod_name, mod in models.items():
        mod_result = []
        print('Training {}'.format(mod_name))
        # Train the model
        start_time = time.time()
        model = mod['model'](sim_size=sim_size, test_size=test_size)
        df_train, df_test = model.train_model(X, y, *list(mod.values())[1:])
        train_time = time.time() - start_time
        print("%s seconds" % train_time)
        if figsize is not None:
            # Plot simulation
            plt.figure(figsize=figsize)
            ax = plt.gca()
            ax = model.create_errorband_axes(ax)
            ax.set_title(mod_name)
            ax.set_ylabel('Accuracy')
            ax.set_xlabel(model.setting)
            if model.log:
                ax.set_xscale('log')
            plt.legend()
            plt.show()

        # Get results
        best_param = model.get_bestparameter()
        accuracy = df_test.mean(axis=1).loc[best_param]
        best_param = f'{model.setting} = {best_param}'
        try:
            top_predictor = model.get_toppredictors().idxmax()[0]
        except NotImplementedError:
            top_predictor = 'NA'
        # Store model results in a list
        mod_result.append(mod_name)
        mod_result.append(best_param)
        mod_result.append(accuracy)
        mod_result.append(top_predictor)
        mod_result.append(train_time)
        # Collect all models results
        data.append(mod_result)
    return pd.DataFrame(data, columns=['Model', 'Best Parameter',
                                       'Accuracy', 'Top Predictor',
                                       'Train Time'])