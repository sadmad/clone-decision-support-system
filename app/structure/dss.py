import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from app import app
from app import scale
from app.structure import model
from app.structure import model_accuracy

# import pickle

app_path = os.path.join(app.root_path, 'models')

features = [
    "Cultivator",
    "Alchol",
    "Malic_Acid",
    "Ash",
    "Alcalinity_of_Ash",
    "Magnesium",
    "Total_phenols",
    "Falvanoids",
    "Nonflavanoid_phenols",
    "Proanthocyanins",
    "Color_intensity",
    "Hue",
    "OD280",
    "Proline"
]


class DSS:
    def __init__(self, obj=None):
        self.train_data = None
        self.x_train = None
        self.y_train = None
        self.model_key = obj['model']
        self.saved_model_scaler = os.path.join(app_path, obj['scaler'])
        self.saved_model_path = os.path.join(app_path, self.model_key)

        self.test_data = None
        self.x_test = None
        self.y_test = None

    def data_intialization(self):
        self.train_data = pd.read_csv('wine_data.csv', names=features)
        self.x_train = self.train_data.drop('Cultivator', axis=1)
        self.y_train = self.train_data['Cultivator']

    def data_preprocessing(self):
        self.x_train = scale.Scale.StandardScaler(self.x_train, self.saved_model_scaler)

    def save_model(self, model=None):
        if os.path.exists(self.saved_model_path):
            os.remove(self.saved_model_path)

        # save the model to disk
        # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

        # pickle.dump(model, open(app.config['NEURAL_NETWORK_MODEL'], 'wb'))
        joblib.dump(model, self.saved_model_path)

    def testing(self):
        self.test_data = pd.read_csv('wine_data_test.csv', names=features)
        self.x_test = self.test_data.drop('Cultivator', axis=1)
        self.y_test = self.test_data['Cultivator']

        self.x_test = scale.Scale.LoadScalerAndScaleTestData(self.x_test, self.saved_model_scaler)

        # load the model from disk
        # loaded_model = pickle.load(open(saved_model_path, 'rb'))
        loaded_model = joblib.load(self.saved_model_path)
        score_result = loaded_model.score(self.x_test, self.y_test)
        predictions = loaded_model.predict(self.x_test)
        # print(confusion_matrix(self.y_test,predictions))
        # print(classification_report(self.y_test,predictions))

        return pd.Series(predictions).to_json(orient='values')
        # return jsonify([{
        #     'status':200,
        #     'message':'Test Obervations are predicted by Neural Network Trained Model.',
        #     'predictions' : pd.Series(predictions).to_json(orient='values')
        # }])

    def gridSearch(self, classifier, grid_param):
        # https://stackabuse.com/cross-validation-and-grid-search-for-model-selection-in-python/
        # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74##targetText=In%20the%20case%20of%20a,each%20node%20learned%20during%20training).
        from sklearn.model_selection import GridSearchCV

        gd_sr = GridSearchCV(estimator=classifier,
                             param_grid=grid_param,
                             scoring='accuracy',
                             cv=5,
                             n_jobs=-1)
        gd_sr.fit(self.x_train, self.y_train)

        best_parameters = gd_sr.best_params_
        print(best_parameters)


class NeuralNetwork(DSS):

    def getClassifier(self):
        return MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)

    def determineBestHyperParameters(self):
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
        super().gridSearch(self.getClassifier(), grid_param)

    def training(self):
        # self.determineBestHyperParameters()

        generalModel = self.getClassifier()
        generalModel.fit(self.x_train, self.y_train)

        modelObject = model.Model()
        modelObject.set_name('NEURAL_NETWORK_MODEL')
        modelObject.set_trained_model(generalModel)
        return model_accuracy.ModelAccuracy.stratified_k_fold(generalModel, self.x_train, self.y_train, modelObject)


class RandomForest(DSS):

    def getClassifier(self):
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=2,
            random_state=0,
            criterion='gini',
            bootstrap=True
        )

    def determineBestHyperParameters(self):
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
        super().gridSearch(self.getClassifier(), grid_param)

    def training(self):
        # X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)

        # self.determineBestHyperParameters()

        generalModel = self.getClassifier()

        generalModel.fit(self.x_train, self.y_train)

        modelObject = model.Model()
        modelObject.set_name('RANDOM_FOREST_CLASSIFIER_MODEL')
        modelObject.set_trained_model(generalModel)
        return model_accuracy.ModelAccuracy.stratified_k_fold(generalModel, self.x_train, self.y_train, modelObject)


class LinearRegressionM(DSS):

    def getClassifier(self):
        return LinearRegression()

    def determineBestHyperParameters(self):
        grid_param = {
            'fit_intercept': [True, False],
            'normalize': [True, False],
            'copy_X': [True, False]
        }
        super().gridSearch(self.getClassifier(), grid_param)

    def training(self):
        # self.determineBestHyperParameters()

        generalModel = self.getClassifier()
        generalModel.fit(self.x_train, self.y_train)

        modelObject = model.Model()
        modelObject.set_name('LINEAR_REGRESSION_MODEL')
        modelObject.set_trained_model(generalModel)
        return model_accuracy.ModelAccuracy.stratified_k_fold(generalModel, self.x_train, self.y_train, modelObject)


class LogisticRegressionM(DSS):

    def getClassifier(self):
        return LogisticRegression()

    def determineBestHyperParameters(self):
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
        super().gridSearch(self.getClassifier(), grid_param)

    def training(self):
        # self.determineBestHyperParameters()

        generalModel = self.getClassifier()

        generalModel.fit(self.x_train, self.y_train)

        modelObject = model.Model()
        modelObject.set_name('LOGISTIC_REGRESSION_MODEL')
        modelObject.set_trained_model(generalModel)

        return model_accuracy.ModelAccuracy.stratified_k_fold(generalModel, self.x_train, self.y_train, modelObject)
