import os

import joblib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from app import app
from app import scale
from app.structure import model
from keras.layers import Input, Concatenate
from keras.models import Model
from app.structure import accuracy_finder as accuracy

import redis
import json


class DSS:
    def __init__(self):
        print(' DSS Constructor')

        pass

    def data_preprocessing(self, finding):
        print(' DSS Data Preprocessing')
        return scale.Scale.StandardScaler(finding.x_train, finding.trained_scaler_path)

    def fit(self, classifier, finding):
        ac = 0  # accuracy.AccuracyFinder.stratified_k_fold(classifier, finding.x_train, finding.y_train)
        if os.path.exists(finding.trained_model_path):
            os.remove(finding.trained_model_path)

        fit_model = classifier.fit(finding.x_train, finding.y_train)
        self.save_model(fit_model, finding)
        return ac

        return

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

        # data = scale.Scale.LoadScalerAndScaleTestData(data, finding.trained_scaler_path)

        loaded_model = joblib.load(finding.trained_model_path)
        # score_result = loaded_model.score(finding.x_train, finding.y_train)

        prediction = loaded_model.predict(data)
        cached_response_variables = json.loads(redis.Redis().get(finding.cache_key))

        res = {}
        i = 0
        for j in cached_response_variables:
            res[j] = round(prediction[0][i], 2)
            i = i + 1
        return json.dumps(str(res))
        # print(confusion_matrix(self.y_test,predictions))
        # print(classification_report(self.y_test,predictions))

        # return pd.Series(predictions).to_json(orient='values')

        # return jsonify([{
        #     'status':200,
        #     'message':'Test Obervations are predicted by Neural Network Trained Model.',
        #     'predictions' : pd.Series(predictions).to_json(orient='values')
        # }])

    def gridSearch(self, classifier, grid_param, finding):
        # https://stackabuse.com/cross-validation-and-grid-search-for-model-selection-in-python/
        # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74##targetText=In%20the%20case%20of%20a,each%20node%20learned%20during%20training).
        from sklearn.model_selection import GridSearchCV

        from sklearn.multioutput import MultiOutputRegressor
        classifier = GridSearchCV(MultiOutputRegressor(classifier), param_grid=grid_param)

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
            return MLPRegressor(hidden_layer_sizes=(13, 13, 13), activation='logistic', random_state=1, max_iter=500)

    def training(self, finding):
        return super().fit(self.getClassifier(finding.is_regression), finding)

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
        return super().fit(self.getClassifier(finding.is_regression), finding)

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
################### Decision Tree  ###########################
#######################################################################
#######################################################################
#######################################################################

class DecisionTreeRegressor(DSS):

    def getClassifier(self, is_regression=0):
        print(' DecisionTreeRegressor Return MODEL')
        from sklearn import tree
        if is_regression == 0:
            return tree.DecisionTreeClassifier()
        else:

            return tree.DecisionTreeRegressor()

    def training(self, finding):
        return super().fit(self.getClassifier(finding.is_regression), finding)

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


#######################################################################
#######################################################################
#######################################################################
################### Deep Neural Network Model #########################
#######################################################################
#######################################################################
#######################################################################

class DeepNeuralNetwork(DSS):

    def getClassifier(self, finding):

        from keras.models import Sequential
        from keras.layers.core import Dense

        from keras import backend as K
        from keras.optimizers import Adam
        import tensorflow as tf
        print(tf.__version__)

        print(' Deep NeuralNetwork  Model')
        K.clear_session()
        model = Sequential()
        columns_x = len(finding.x_train[0])
        # columns_x = len(finding.x_train[0])
        columns_y = len(finding.y_train.columns)
        output_neuron_c = 3
        # output_neuron_r = columns_yasaaaaaaaaaaaaaa
        activation_function = 'relu'
        output_activation_function_r = 'relu'
        output_activation_function__c = 'relu'
        optimizer = Adam(lr=0.05)

        if columns_x is not None:
            neuron_count = columns_x
            model.add(Dense(neuron_count, input_dim=columns_x, activation=activation_function))
            hidden_layers = columns_x

            output_neuron_r = columns_y
            for x in range(hidden_layers):
                model.add(Dense(neuron_count, input_dim=columns_x, activation='relu'))
        if finding.is_regression == 0:
            # Classification
            model.add(Dense(output_neuron_c, activation=output_activation_function__c))
            print(len(model.layers))

            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer, metrics=['accuracy'], )
            return model
        else:
            # Regression
            # print(len(model.layers))
            # model.add(Dense(output_neuron_r, activation=output_activation_function_r))
            # print(len(model.layers))
            #
            # model.compile(loss='mse',
            #               optimizer=optimizer, metrics=['accuracy'], )
            # model.summary()


            # Output_Layer = 2#len(finding.y_train[1])
            Output_Layer = len(set(finding.y_train))
            Input_layer = Input(shape=(columns_x,))
            Dense_Layers = Dense(500, activation='relu')(Input_layer)
            Dense_Layers = Dense(256, activation='relu')(Dense_Layers)
            Dense_Layers = Dense(128, activation='relu')(Dense_Layers)
            Dense_Layers = Dense(500, activation='relu')(Dense_Layers)
            Dense_Layers = Dense(1000, activation='relu')(Dense_Layers)

            # Dense_Layers = Concatenate()([Input_layer, Dense_Layers])
            Dense_Layers = Dense(Output_Layer, activation='relu')(Dense_Layers)
            modelReg = Model(inputs=Input_layer, outputs=Dense_Layers)

            # out1 = Dense(1)(Dense_Layers)
            # out2 = Dense(1)(Dense_Layers)

            # modelReg = Model(inputs = Input_layer, outputs = [out1,out2])
            from keras import metrics
            modelReg.compile(
                loss = 'mean_squared_error',
                optimizer = optimizer,
                metrics=[metrics.mean_squared_error,
                         metrics.mean_absolute_error,
                         metrics.mean_absolute_percentage_error]
                # metrics = ['accuracy']
            )
            modelReg.summary()
            print(len(modelReg.layers))
            return modelReg

        # return model1

    def training(self, finding):

        return self.fit(self.getClassifier(finding), finding)

    def fit(self, classifier, finding):

        if finding.is_regression == 0:
            # Classification
            import numpy as np
            B = finding.y_train.transpose()
            B = np.reshape(finding.y_train, (-1, 1))
            encoder = OneHotEncoder()
            targets = encoder.fit_transform(B)
            train_features, test_features, train_targets, test_targets = train_test_split(finding.x_train, targets,
                                                                                          test_size=0.2)

            fit_model = classifier.fit(finding.x_train, targets, epochs=10, batch_size=200, verbose=2)
            # results= fit_model.fit_model.evaluate(test_features,test_targets)
            # print("Accuracy on the test dataset:%.2f" % results[1])
        else:
            # Regression
            # K.clear_session()
            classifier.fit(finding.x_train, finding.y_train, epochs=500)

        self.save_model(classifier, finding)
        return 0

    def save_model(self, model, finding):
        print(' DSS Save Model')
        import shutil

        if os.path.exists(finding.trained_model_path):
            # Force deletion of a file set it to normal
            #os.remove(finding.trained_model_path)
            shutil.rmtree(finding.trained_model_path)
        model.save(finding.trained_model_path)

        # joblib.dump(model, finding.trained_model_path)

    def predict_data(self, finding, data):
        print(' DSS predict_data')

        import keras
        from keras.models import Sequential
        from keras.layers.core import Dense
        from keras import backend as K
        from keras.optimizers import Adam

        K.clear_session()
        # data = scale.Scale.LoadScalerAndScaleTestData(data, finding.trained_scaler_path)

        # loaded_model = joblib.load(finding.trained_model_path)

        reconstructed_model = keras.models.load_model(finding.trained_model_path)
        predictions = reconstructed_model.predict(data)
        # predictions = loaded_model.model.predict(data)
        # Awais
        # predictions = loaded_model.model.predict(data)
        # import tensorflow as tf
        # global graph
        # graph = tf.get_default_graph()
        # with graph.as_default():
        #     predictions = loaded_model.model.predict(data, batch_size=1, verbose=1)




        cached_response_variables = json.loads(redis.Redis().get(finding.cache_key))
        # Something went wrong in Cache for response variable
        res = {}
        i = 0
        for j in cached_response_variables:
            res[j] = round(predictions[0][i], 2)
            i = i + 1
        return json.dumps(str(res))
        # return predictions[0] #pd.Series(predictions).to_json(orient='values')

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
        super().gridSearch(self.getClassifier(), grid_param, finding, model)
