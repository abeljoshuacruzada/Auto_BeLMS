import numpy as np
import pandas as pd
import warnings
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.exceptions import ConvergenceWarning
from tqdm.autonotebook import tqdm
from .MyMLModels import MyMLModels


class MyDecisionTree(MyMLModels):
    """Represents a Decision Tree Classifier model"""
    
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
        self.setting = 'max depth'
        self.max_features = None
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.splitter = 'best'
        self.random_state = None
        self.tree = None
        self.n_features = None
        self.n_outputs = None
        