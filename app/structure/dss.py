import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

from app import app
from app import scale
from app.structure import model
from app.structure import accuracy_finder as accuracy


class DSS:
    def __init__(self):
        print(' DSS Constructor')

        pass

    def data_preprocessing(self, finding):
        print(' DSS Data Preprocessing')
        return scale.Scale.StandardScaler(finding.x_train, finding.trained_scaler_path)

    def fit(self, classifier, finding):
        fit_model = classifier.fit(finding.x_train, finding.y_train)
        self.save_model(fit_model, finding)
        return 0
        # return accuracy.AccuracyFinder.stratified_k_fold(fit_model, finding.x_train, finding.y_train)

    def save_model(self, model, finding):
        print(' DSS Save Model')
        if os.path.exists(finding.trained_model_path):
            os.remove(finding.trained_model_path)
        joblib.dump(model, finding.trained_model_path)

    def testing(self, finding):
        print(' DSS testing')

        finding.x_train = scale.Scale.LoadScalerAndScaleTestData(finding.x_train, finding.trained_scaler_path)

        loaded_model = joblib.load(finding.trained_model_path)
        # score_result = loaded_model.score(finding.x_train, finding.y_train)
        predictions = loaded_model.predict(finding.x_train)
        # print(confusion_matrix(self.y_test,predictions))
        # print(classification_report(self.y_test,predictions))

        return pd.Series(predictions).to_json(orient='values')
        # return jsonify([{
        #     'status':200,
        #     'message':'Test Obervations are predicted by Neural Network Trained Model.',
        #     'predictions' : pd.Series(predictions).to_json(orient='values')
        # }])

    def predict_data(self, finding, data):
        print(' DSS predict_data')

        data = scale.Scale.LoadScalerAndScaleTestData(data, finding.trained_scaler_path)

        loaded_model = joblib.load(finding.trained_model_path)
        # score_result = loaded_model.score(finding.x_train, finding.y_train)
        predictions = loaded_model.predict(data)
        # print(confusion_matrix(self.y_test,predictions))
        # print(classification_report(self.y_test,predictions))

        return pd.Series(predictions).to_json(orient='values')
        # return jsonify([{
        #     'status':200,
        #     'message':'Test Obervations are predicted by Neural Network Trained Model.',
        #     'predictions' : pd.Series(predictions).to_json(orient='values')
        # }])

    def gridSearch(self, classifier, grid_param, finding):
        # https://stackabuse.com/cross-validation-and-grid-search-for-model-selection-in-python/
        # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74##targetText=In%20the%20case%20of%20a,each%20node%20learned%20during%20training).
        from sklearn.model_selection import GridSearchCV

        gd_sr = GridSearchCV(estimator=classifier,
                             param_grid=grid_param,
                             scoring='accuracy',
                             cv=5,
                             n_jobs=-1)
        gd_sr.fit(finding.x_train, finding.y_train)
        best_parameters = gd_sr.best_params_
        print(best_parameters)


#######################################################################
#######################################################################
#######################################################################
################### Neural Network Model ##############################
#######################################################################
#######################################################################
#######################################################################

class NeuralNetwork(DSS):

    def getClassifier(self, is_regression=0):
        print(' NeuralNetwork Return Model')
        if is_regression == 0:
            return MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
        else:
            return MLPRegressor(hidden_layer_sizes=(13, 13, 13), max_iter=500)

    def training(self, finding):
        return super().fit(self.getClassifier(finding.regression), finding)

    def determineBestHyperParameters(self, finding):
        grid_param = {
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            # 'shuffle': [True,False],
            # 'verbose': [True,False],
            # 'warm_start': [True,False],
            # 'nesterovs_momentum': [True,False],
            # 'early_stopping': [True,False]
        }
        super().gridSearch(self.getClassifier(), grid_param, finding)


#######################################################################
#######################################################################
#######################################################################
################### Random Forest Model ###############################
#######################################################################
#######################################################################
#######################################################################

class RandomForest(DSS):

    def getClassifier(self, is_regression=0):
        print(' RandomForest Return Model ')
        if is_regression == 0:
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=2,
                random_state=0,
                criterion='gini',
                bootstrap=True
            )
        else:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=2,
                random_state=0,
                criterion='mse',
                bootstrap=True
            )

    def training(self, finding):
        return super().fit(self.getClassifier(finding.regression), finding)

    def determineBestHyperParameters(self, finding):
        grid_param = {
            'criterion': ['gini', 'entropy'],
            'bootstrap': [True, False],

            # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            # 'max_features': ['auto', 'sqrt'],
            # 'min_samples_leaf': [1, 2, 4],
            # 'min_samples_split': [2, 5, 10],
            # 'n_estimators': [100,200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
            'n_estimators': [100]

        }
        super().gridSearch(self.getClassifier(), grid_param, finding)


#######################################################################
#######################################################################
#######################################################################
################### Linear Regression Model ###########################
#######################################################################
#######################################################################
#######################################################################

class LinearRegressionM(DSS):

    def getClassifier(self):
        print(' LinearRegressionM Return MODEL')
        return LinearRegression()

    def training(self, finding):
        return super().fit(self.getClassifier(), finding)

    def determineBestHyperParameters(self, finding):
        grid_param = {
            'fit_intercept': [True, False],
            'normalize': [True, False],
            'copy_X': [True, False]
        }
        super().gridSearch(self.getClassifier(), grid_param, finding)


#######################################################################
#######################################################################
#######################################################################
################### Logistic Regression Model #########################
#######################################################################
#######################################################################
#######################################################################

class LogisticRegressionM(DSS):

    def getClassifier(self):
        print(' LogisticRegressionM Return Model')
        return LogisticRegression()

    def training(self, finding):
        return super().fit(self.getClassifier(), finding)

    def determineBestHyperParameters(self, finding):
        grid_param = {
            # 'penalty': ['l1','l2','elasticnet','none'],
            # 'penalty': ['l2','elasticnet','none'],

            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            # 'dual':[True, False],
            'fit_intercept': [True, False],
            # 'class_weight':[dict, 'balanced',None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            # 'multi_class':['ovr', 'multinomial','auto'],
            'warm_start': ['True', 'False']
        }
        super().gridSearch(self.getClassifier(), grid_param, finding)
