import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from app import app
from app import scale
from app.structure import accuracy_finder as accuracy
import redis
import json


class DSS:

    def __init__(self):
        pass

    def data_preprocessing(self, data):
        return scale.Scale.StandardScaler(data.x_train, data.trained_scaler_path)

    def fit(self, model, data):
        ac = 0  # accuracy.AccuracyFinder.stratified_k_fold(classifier, finding.x_train, finding.y_train)
        if os.path.exists(data.trained_model_path):
            os.remove(data.trained_model_path)

        fit_model = model.fit(data.x_train, data.y_train)
        self.save_model(fit_model, data)
        return ac

    def evaluate_accuracy(self, model, data):
        ac = accuracy.AccuracyFinder.stratified_k_fold(model, data.x_train, data.y_train)
        return ac

    def evaluate_accuracy_dnn(self, model, data):
        ac = accuracy.AccuracyFinder.stratified_k_fold_dnn(model, data.x_train, data.y_train, data)
        return ac

    def save_model(self, model, data):
        if os.path.exists(data.trained_model_path):
            os.remove(data.trained_model_path)
        joblib.dump(model, data.trained_model_path)

    def predict_data(self, data):
        loaded_model = joblib.load(data.trained_model_path)
        # score_result = loaded_model.score(finding.x_train, finding.y_train)
        prediction = loaded_model.predict(data.test_data)
        cached_response_variables = json.loads(redis.Redis().get(data.cache_key))
        res = {}
        i = 0
        for j in cached_response_variables:
            res[j] = round(prediction[0][i], 2)
            i = i + 1
        return res

    def grid_search(self, model, grid_param, data):
        # https://stackabuse.com/cross-validation-and-grid-search-for-model-selection-in-python/
        # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74##targetText=In%20the%20case%20of%20a,each%20node%20learned%20during%20training).
        from sklearn.model_selection import GridSearchCV
        from sklearn.multioutput import MultiOutputRegressor
        classifier = GridSearchCV(MultiOutputRegressor(model), param_grid=grid_param)
        gd_sr = GridSearchCV(estimator=classifier,
                             param_grid=grid_param,
                             scoring='accuracy',
                             cv=5,
                             n_jobs=-1)
        gd_sr.fit(data.x_train, data.y_train)
        best_parameters = gd_sr.best_params_

    def dummy_regressor(self, data):

        from sklearn.dummy import DummyRegressor
        from sklearn.model_selection import cross_val_score, RepeatedKFold
        from numpy import absolute, mean, std
        from sklearn import model_selection
        if data.is_regression == 1:

            rfr = DummyRegressor(strategy='median').fit(data.x_train, data.y_train)
            scoring = 'neg_mean_absolute_error'
            # scoring = 'neg_mean_squared_error'
            # scoring = 'r2'
            kfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            results = cross_val_score(rfr, data.x_train, data.y_train, scoring=scoring, cv=kfold, n_jobs=-1)
            n_scores = absolute(results)
            print("DummyRegressor: %.3f (%.3f)" % (mean(n_scores), std(n_scores)))
            return mean(n_scores)

            # from numpy import sqrt
            # from sklearn.metrics import mean_squared_error
            # from sklearn.metrics import mean_absolute_errorcross_validation
            # X_train, X_test, y_train, y_test = model_selection.train_test_split(
            #     data.x_train, data.y_train, test_size=0.25, random_state=42, shuffle=True)
            #
            # # Create a dummy regressor
            # dummy = DummyRegressor(strategy='mean')
            #
            # # Train dummy regressor
            # dummy.fit(X_train, y_train)
            # y_true, y_pred = y_test, dummy.predict(X_test)
            # # Dummy performance
            # mean_absolute_error = mean_absolute_error(y_test, y_pred)
            # mean_squared_error = sqrt(mean_squared_error(y_test, y_pred))
            # print("mean_squared_error: %.3f (%.3f)" % (mean_absolute_error, mean_squared_error))
            # print("**** Done ****")
        else:
            # NotImplemented
            pass
        return 0

    def determine_accuracy(self, data, model):
        from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold
        from numpy import absolute, mean, std
        scoring = 'neg_mean_absolute_error'
        # scoring = 'neg_mean_squared_error'
        # scoring = 'r2'
        kfold = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
        results = cross_val_score(model, data.x_train, data.y_train, scoring=scoring, cv=kfold, n_jobs=-1)
        n_scores = absolute(results)
        print("MAE: %.3f (%.3f)" % (mean(n_scores), std(n_scores)))
        return mean(n_scores)


#######################################################################
#######################################################################
#######################################################################
################### Neural Network Model ##############################
#######################################################################
#######################################################################
#######################################################################

class NeuralNetwork(DSS):

    def __init__(self):
        self.model_name = app.config['NEURAL_NETWORK_MODEL']['name']
        self.model_file_name = app.config['NEURAL_NETWORK_MODEL']['model']
        self.scaler_file_name = app.config['NEURAL_NETWORK_MODEL']['scaler']
        pass

    def get_model(self, is_regression=0):
        if is_regression == 0:
            return MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
        else:
            return MLPRegressor(hidden_layer_sizes=(13, 13, 13), activation='logistic', random_state=1, max_iter=500)

    def training(self, data):
        return super().fit(self.get_model(data.is_regression), data)

    def determine_best_hyper_parameters(self, data):
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
        super().grid_search(self.get_model(), grid_param, data)

    def get_model_grid_search(self, data):

        from sklearn.model_selection import GridSearchCV
        if data.is_regression == 1:
            param_grid = {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100, 1)],
                          'activation': ['identity', 'relu', 'tanh', 'logistic'],
                          'alpha': [0.0001, 0.05],
                          # 'learning_rate': ['constant', 'adaptive'],
                          'solver': ['adam'],
                          }

            gsc = GridSearchCV(
                estimator=MLPRegressor(),
                param_grid=param_grid,
                cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

            grid_result = gsc.fit(data.x_train, data.y_train)
            best_params = grid_result.best_params_

            rfr = MLPRegressor(
                hidden_layer_sizes=best_params["hidden_layer_sizes"],
                activation=best_params["activation"],
                alpha=best_params["alpha"],
                # learning_rate=best_params["learning_rate"],
                solver=best_params["solver"],
                max_iter=5000,
                n_iter_no_change=200
            )

            return self.determine_accuracy(data, rfr)

        else:
            # NotImplemented
            pass
        return 0

    def accuracy_evaluation(self, data):
        # return self.dummy_regressor(data)
        return self.get_model_grid_search(data)
        # return super().evaluate_accuracy(self.get_model(data), data)


#######################################################################
#######################################################################
#######################################################################
################### Random Forest Model ###############################
#######################################################################
#######################################################################
#######################################################################

class RandomForest(DSS):

    def __init__(self):
        self.model_name = app.config['RANDOM_FOREST_MODEL']['name']
        self.model_file_name = app.config['RANDOM_FOREST_MODEL']['model']
        self.scaler_file_name = app.config['RANDOM_FOREST_MODEL']['scaler']

    def get_model(self, is_regression=0):
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

    def training(self, data):
        model = self.get_model(data.is_regression) if app.config['GRID_SEARCH'] == 0 else self.get_model_grid_search(
            data)
        return super().fit(model, data)

    def determine_best_hyper_parameters(self, data):
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
        super().grid_search(self.get_model(), grid_param, data)

    def get_model_grid_search(self, data):

        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold
        from numpy import absolute, mean, std

        if data.is_regression == 1:
            param_grid = {
                'max_depth': range(3, 10),
                'n_estimators': (10, 50, 100, 1000),
            }
            gsc = GridSearchCV(
                estimator=RandomForestRegressor(),
                param_grid=param_grid,
                cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

            grid_result = gsc.fit(data.x_train, data.y_train)
            best_params = grid_result.best_params_
            rfr = RandomForestRegressor(
                max_depth=best_params["max_depth"],
                n_estimators=best_params["n_estimators"],
                random_state=False,
                verbose=False
            )
            return self.determine_accuracy(data, rfr)
        else:
            # NotImplemented
            pass
        return 0

    def accuracy_evaluation(self, data):
        return self.dummy_regressor(data)
        # return self.get_model_grid_search(data)
        # return super().evaluate_accuracy(self.get_model(data), data)


#######################################################################
#######################################################################
#######################################################################
################### Linear Regression Model ###########################
#######################################################################
#######################################################################
#######################################################################

class LinearRegressionM(DSS):

    def __init__(self):
        self.model_name = app.config['LINEAR_REGRESSION_MODEL']['name']
        self.model_file_name = app.config['LINEAR_REGRESSION_MODEL']['model']
        self.scaler_file_name = app.config['LINEAR_REGRESSION_MODEL']['scaler']
        pass

    def get_model(self):
        return LinearRegression()

    def training(self, data):
        return super().fit(self.get_model(), data)

    def determine_best_hyper_parameters(self, data):
        grid_param = {
            'fit_intercept': [True, False],
            'normalize': [True, False],
            'copy_X': [True, False]
        }
        super().grid_search(self.get_model(), grid_param, data)

    def get_model_grid_search(self, data):

        from sklearn.linear_model import LinearRegression

        from sklearn.linear_model import ElasticNet
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold
        from numpy import absolute, std, mean

        if data.is_regression == 1:

            # For LinearRegression
            # param_grid = {
            #     'fit_intercept': [True, False],
            #     'normalize': [True, False],
            #     "copy_X": [True, False],
            #     "solver": ['svd', 'cholesky', 'lsqr', 'sag']
            #     # "positive": [True, False],
            # }

            # gsc = GridSearchCV(
            #     estimator=LinearRegression(),
            #     param_grid=param_grid,
            #     cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
            #
            # grid_result = gsc.fit(data.x_train, data.y_train)
            # best_params = grid_result.best_params_
            # rfr = LinearRegression(fit_intercept=best_params["fit_intercept"], normalize=best_params["normalize"],
            #                        copy_X=best_params["copy_X"], n_jobs=-1, solver=best_params["solver"])

            # scores = cross_val_score(rfr, data.x_train, data.y_train, cv=10, scoring='neg_mean_absolute_error')
            # print("MAE: %.3f (%.3f)" % (scores))

            # For ElasticNet
            param_grid = {
                'alpha': [1, 3, 5, 7],
                'fit_intercept': [True, False],
                "copy_X": [True, False],
                "selection": ['cyclic', 'random']
                # "positive": [True, False],
            }

            gsc = GridSearchCV(
                estimator=ElasticNet(),
                param_grid=param_grid,
                cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

            grid_result = gsc.fit(data.x_train, data.y_train)
            best_params = grid_result.best_params_
            rfr = ElasticNet(
                alpha=best_params["alpha"],
                fit_intercept=best_params["fit_intercept"],
                copy_X=best_params["copy_X"],
                selection=best_params["selection"]
            )
            return self.determine_accuracy(data, rfr)
        else:
            # NotImplemented
            pass
        return 0

    def accuracy_evaluation(self, data):
        return self.dummy_regressor(data)
        # return self.get_model_grid_search(data)
        # return super().evaluate_accuracy(self.get_model(data), data)


#######################################################################
#######################################################################
#######################################################################
################### Decision Tree  ###########################
#######################################################################
#######################################################################
#######################################################################

class DecisionTree(DSS):

    def __init__(self):
        self.model_name = app.config['DECISION_TREE_MODEL']['name']
        self.model_file_name = app.config['DECISION_TREE_MODEL']['model']
        self.scaler_file_name = app.config['DECISION_TREE_MODEL']['scaler']
        pass

    def get_model(self, is_regression=0):
        from sklearn import tree
        if is_regression == 0:
            return tree.DecisionTreeClassifier()
        else:

            return tree.DecisionTreeRegressor()

    def training(self, data):
        return super().fit(self.get_model(data.is_regression), data)

    def determine_best_hyper_parameters(self, data):
        grid_param = {
            'fit_intercept': [True, False],
            'normalize': [True, False],
            'copy_X': [True, False]
        }
        super().grid_search(self.get_model(), grid_param, data)

    def get_model_grid_search(self, data):

        from sklearn import tree
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold
        from numpy import absolute, mean, std

        if data.is_regression == 1:
            param_grid = {
                'criterion': ["mse", "friedman_mse", "mae"],
                'splitter': ["best", "random"],
                "max_features": ["auto", "sqrt", "log2"]
            }

            gsc = GridSearchCV(
                estimator=tree.DecisionTreeRegressor(),
                param_grid=param_grid,
                cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

            grid_result = gsc.fit(data.x_train, data.y_train)
            best_params = grid_result.best_params_
            rfr = tree.DecisionTreeRegressor(criterion=best_params["criterion"], splitter=best_params["splitter"],
                                             max_features=best_params["max_features"])

            return self.determine_accuracy(data, rfr)
        else:
            # NotImplemented
            pass
        return 0

    def accuracy_evaluation(self, data):
        return self.dummy_regressor(data)
        # return self.get_model_grid_search(data)
        # return super().evaluate_accuracy(self.get_model(data), data)


#######################################################################
#######################################################################
#######################################################################
################### Deep Neural Network Model #########################
#######################################################################
#######################################################################
#######################################################################

class DeepNeuralNetwork(DSS):

    def __init__(self):
        self.model_name = app.config['DEEP_NEURAL_NETWORK_MODEL']['name']
        self.model_file_name = app.config['DEEP_NEURAL_NETWORK_MODEL']['model']
        self.scaler_file_name = app.config['DEEP_NEURAL_NETWORK_MODEL']['scaler']
        pass

    def get_model(self, finding):
        from tensorflow import keras
        # from keras.models import Sequential
        # from keras.layers.core import Dense

        from keras import backend as K
        # from keras.optimizers import Adam
        # from keras.layers import Input
        # from keras.models import Model

        # K.clear_session()

        columns_x = len(finding.x_train[0])
        # columns_x = len(finding.x_train[0])
        columns_y = len(finding.y_train.columns)
        output_neuron_c = 3
        # output_neuron_r = columns_yasaaaaaaaaaaaaaa
        activation_function = 'relu'
        output_activation_function_r = 'relu'
        output_activation_function__c = 'relu'
        optimizer = keras.optimizers.Adam(lr=0.05)

        if finding.is_regression == 0:
            # Classification

            if columns_x is not None:
                model = keras.Sequential()
                neuron_count = columns_x
                model.add(keras.layers.Dense(neuron_count, input_dim=columns_x, activation=activation_function))
                hidden_layers = columns_x

                output_neuron_r = columns_y
                for x in range(hidden_layers):
                    model.add(keras.layers.Dense(neuron_count, input_dim=columns_x, activation='relu'))

            model.add(keras.layers.Dense(output_neuron_c, activation=output_activation_function__c))

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
            Input_layer = keras.Input(shape=(columns_x,))
            Dense_Layers = keras.layers.Dense(500, activation='relu')(Input_layer)
            Dense_Layers = keras.layers.Dense(256, activation='relu')(Dense_Layers)
            Dense_Layers = keras.layers.Dense(128, activation='relu')(Dense_Layers)
            Dense_Layers = keras.layers.Dense(500, activation='relu')(Dense_Layers)
            Dense_Layers = keras.layers.Dense(1000, activation='relu')(Dense_Layers)

            # Dense_Layers = Concatenate()([Input_layer, Dense_Layers])
            Dense_Layers = keras.layers.Dense(Output_Layer, activation='linear')(Dense_Layers)
            modelReg = keras.Model(inputs=Input_layer, outputs=Dense_Layers)

            # out1 = Dense(1)(Dense_Layers)
            # out2 = Dense(1)(Dense_Layers)

            # modelReg = Model(inputs = Input_layer, outputs = [out1,out2])
            from keras import metrics
            modelReg.compile(
                loss='mean_squared_error',
                optimizer=optimizer,
                metrics=[metrics.mean_squared_error,
                         metrics.mean_absolute_error,
                         metrics.mean_absolute_percentage_error]
                # metrics = ['accuracy']
            )
            # modelReg.summary()
            return modelReg

        # return model1

    def fit(self, model, data):
        if data.is_regression == 0:
            # Classification
            import numpy as np
            B = data.y_train.transpose()
            B = np.reshape(data.y_train, (-1, 1))
            encoder = OneHotEncoder()
            targets = encoder.fit_transform(B)
            train_features, test_features, train_targets, test_targets = train_test_split(data.x_train, targets,
                                                                                          test_size=0.2)

            fit_model = model.fit(data.x_train, targets, epochs=10, batch_size=200, verbose=2)
            # results= fit_model.fit_model.evaluate(test_features,test_targets)
            # print("Accuracy on the test dataset:%.2f" % results[1])
        else:
            # Regression
            # K.clear_session()
            # verbose=2 , 0 for not printing epocs
            model.fit(data.x_train, data.y_train, epochs=500, verbose=0)

        self.save_model(model, data)
        return 0

    def save_model(self, model, finding):
        import shutil
        if os.path.exists(finding.trained_model_path):
            shutil.rmtree(finding.trained_model_path)
        model.save(finding.trained_model_path)

    def training(self, data):
        return self.fit(self.get_model(data), data)

    def accuracy_evaluation(self, data):
        return super().evaluate_accuracy_dnn(self.get_model(data), data)

    def predict_data(self, data):
        from tensorflow import keras
        # from keras.models import Sequential
        # from keras.layers.core import Dense
        # from keras import backend as K
        # from keras.optimizers import Adam

        # K.clear_session()
        # data = scale.Scale.LoadScalerAndScaleTestData(data, finding.trained_scaler_path)

        # loaded_model = joblib.load(finding.trained_model_path)

        reconstructed_model = keras.models.load_model(data.trained_model_path)
        predictions = reconstructed_model.predict(data.test_data)
        # predictions = loaded_model.model.predict(data)
        # Awais
        # predictions = loaded_model.model.predict(data)
        # import tensorflow as tf
        # global graph
        # graph = tf.get_default_graph()
        # with graph.as_default():
        #     predictions = loaded_model.model.predict(data, batch_size=1, verbose=1)

        cached_response_variables = json.loads(redis.Redis().get(data.cache_key))

        result = pd.np.array(predictions.tolist())
        # Something went wrong in Cache for response variable
        res = {}
        i = 0
        for j in cached_response_variables:
            res[j] = round(result[0][i], 2)
            i = i + 1

        return res
        # return json.dumps(str(res))
        # return predictions[0] #pd.Series(predictions).to_json(orient='values')

    def determine_best_hyper_parameters(self, data):
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
        super().grid_search(self.self.get_model(data), grid_param, data)
