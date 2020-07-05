# Auto-BeLMS
Auto Beneficial Learning Model System

Version 2.2

New Feature

	* Added LinearRegression (Ridge and Lasso)
	* create_weightplot_axes method for LinearModels to plot weights of each parameter
	* simulate_regressor method for AutoBelms
	* new model structure, AutoBelms.regression_models

Other Changes:

	* AutoBelms.models renamed to classifier_models
	* AutoBelms.simulate renamed to AutoBelms.simulate_classifiers
	* AutoBelms.classifier_models['njobs'] renamed to AutoBelms.classifier_models['n_jobs']
	* Fix bug for KNN and SVM get_toppredictors
	* Fix bug for AutoBelms simulate_classifiers

Note:
To run Example.ipynb, place the notebook outside Auto_BeLMS directory
