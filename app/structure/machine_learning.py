import json
import os
import os.path
from os import path

import joblib
import redis
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app import app
from app.structure import data_transformer as dt, factory


class MachineLearning:
    def __init__(self, model_type=None, action_id=None, protection_goods_id=None, user_id=None):
        self.DSS = factory.ModelFactory().get_model(model_type)
        self.model_id = model_type
        self.action_id = action_id
        self.protection_goods_id = protection_goods_id
        self.data = None
        self.input_variables = None
        self.output_variables = None
        self.x_train = None
        self.y_train = None
        self.is_regression = 1
        self.test_data = None
        self.user_id = user_id
        self.cache_key = str(self.user_id) + '_' + str(self.action_id) + '_' + str(self.protection_goods_id)
        self.scaler_file_path = os.path.join(app.config['STORAGE_DIRECTORY'],
                                             'scaler_' + str(self.user_id) + str(self.action_id) +
                                             str(self.protection_goods_id) + '_' + self.DSS.scaler_file_name)

        self.trained_model_path = os.path.join(app.config['STORAGE_DIRECTORY'],
                                               str(self.user_id) + str(self.action_id) +
                                               str(self.protection_goods_id) + '_' + self.DSS.model_file_name)

    def process(self):

        # https://scikit-learn.org/stable/modules/tree.html

        if not path.exists(app.config['STORAGE_DIRECTORY']):
            os.mkdir(app.config['STORAGE_DIRECTORY'])
        self.data_load()
        if self.data.empty:
            return {
                'status': 502,
                'message': 'Data Not Found'
            }

        self.data_intialization()

        # from sklearn.datasets import make_regression
        # self.y_train = make_regression(n_samples=2000, n_features=10, n_informative=8, n_targets=2,
        #                                random_state=1)
        self.data_preprocessing()

        self.training()
        self.training_history_log()
        return {
            'status': 200,
            'message': 'Success'
        }

    def accuracy_finder(self):

        # https://scikit-learn.org/stable/modules/tree.html

        if not path.exists(app.config['STORAGE_DIRECTORY']):
            os.mkdir(app.config['STORAGE_DIRECTORY'])
        self.data_load()
        if self.data.empty:
            return {
                'status': 502,
                'message': 'Data Not Found'
            }

        self.data_intialization()

        # from sklearn.datasets import make_regression
        # self.y_train = make_regression(n_samples=2000, n_features=10, n_informative=8, n_targets=2,
        #                                random_state=1)
        self.data_preprocessing()

        ac = self.accuracy()
        # self.training_history_log()
        return {
            'status': 200,
            'message': 'Success',
            'accuracy': ac
        }

    def data_load(self):
        amucad = dt.Amucad()
        self.data, self.input_variables, self.output_variables = amucad.amucad_generic_api(self)

        # Write data to CSV File
        if app.config['CACHE_API'] == 1:
            fileName = os.path.join(app.config['STORAGE_DIRECTORY'],
                                    str(self.action_id) +
                                    str(self.protection_goods_id) + '_' + "dynamic_data.csv")
            self.data.to_csv(fileName, index=False)

    def data_intialization(self):

        self.data.round(2)

        # drop nulls from response variable's columns
        for y in self.output_variables:
            self.data = self.data.dropna(how='any', subset=[y])

        self.x_train = self.data[self.input_variables]

        # Separation of output variables
        self.y_train = self.data[self.output_variables]

    def data_preprocessing(self, scaling=True):

        # Numeric Imputation
        impute_numerical = SimpleImputer(strategy="mean")
        numerical_transformer = Pipeline(
            steps=[('impute_numerical', impute_numerical)])
        self.x_train = ColumnTransformer(transformers=[('numerical', numerical_transformer, self.input_variables)],
                                         remainder="passthrough").fit_transform(self.x_train)
        if scaling:
            self.data_scaling()

    def data_scaling(self):
        # Data Scaling
        scaler = StandardScaler()
        scaler.fit(self.x_train)
        if os.path.exists(self.scaler_file_path):
            os.remove(self.scaler_file_path)
        joblib.dump(scaler, self.scaler_file_path)
        self.x_train = scaler.transform(self.x_train)

    def training(self):
        r = redis.Redis()
        r.delete(self.cache_key)
        r.mset({self.cache_key: json.dumps(self.output_variables)})
        return self.DSS.training(self)

    def accuracy(self):
        return self.DSS.accuracy_evaluation(self)

    def set_test_data(self, data):
        self.test_data = [data]

    def testing(self):
        if os.path.exists(self.scaler_file_path) and os.path.exists(self.trained_model_path):
            # Before prediction
            self.apply_existing_scaler()
            return self.DSS.predict_data(self)
        else:
            return None

    def apply_existing_scaler(self):
        scaler = joblib.load(self.scaler_file_path)
        self.test_data = scaler.transform(self.test_data)

    def training_history_log(self):

        import datetime
        from app.commons.mongo_connector import MongoConnector
        # https://api.mongodb.com/python/current/tutorial.html
        collection_training = MongoConnector.get_logsdb()
        num_rows_x, num_cols_x = self.x_train.shape
        item = {
            "user_id": 2,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "action_id": self.action_id,
            "protection_goods_id": self.protection_goods_id,
            "training_observations": num_rows_x,
            "input_features": self.input_variables,
            "output_variables": self.output_variables,
            "date": datetime.datetime.utcnow()}
        collection_training.insert_one(item)

    def features_importance(self):

        # https://scikit-learn.org/stable/modules/tree.html
        from sklearn.ensemble import RandomForestRegressor
        self.data_load()
        if self.data.empty:
            return {
                'status': 502,
                'message': 'Data Not Found'
            }

        self.data_intialization()
        # Turn Scaling off
        self.data_preprocessing(False)
        # define the model
        model = RandomForestRegressor()
        # fit the model
        model.fit(self.x_train, self.y_train)
        # get importance
        importance = model.feature_importances_

        # summarize feature importance
        # for i, v in enumerate(importance):
        #     print('Feature: %0d, Score: %.5f' % (i, v))
        # # plot feature importance
        # from matplotlib import pyplot
        # pyplot.bar([x for x in range(len(importance))], importance)
        # pyplot.show()

        data = {}
        j = 0
        for i, v in enumerate(importance):
            data[self.input_variables[j]] = v
            j = j + 1
        return data
