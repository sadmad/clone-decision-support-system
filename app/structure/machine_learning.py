import os
import os.path
from os import path
import random as random
import pandas as pd
from sklearn import preprocessing
from app.structure import data_transformer as dt, dss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from app import app
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import redis
import json


def setDssNetwork(model_type):
    md = model_config = model_name = None

    if model_type == 1:
        md = dss.NeuralNetwork()
        model_config = app.config['NEURAL_NETWORK_MODEL']
        model_name = 'NEURAL_NETWORK_MODEL'

    elif model_type == 2:
        md = dss.RandomForest()
        model_config = app.config['RANDOM_FOREST_CLASSIFIER_MODEL']
        model_name = 'RANDOM_FOREST_CLASSIFIER_MODEL'

    elif model_type == 3:
        md = dss.LinearRegressionM()
        model_config = app.config['LINEAR_REGRESSION_MODEL']
        model_name = 'LINEAR_REGRESSION_MODEL'

    elif model_type == 4:
        md = dss.LogisticRegressionM()
        model_config = app.config['LOGISTIC_REGRESSION_MODEL']
        model_name = 'LOGISTIC_REGRESSION_MODEL'
    elif model_type == 5:
        md = dss.DeepNeuralNetwork()
        model_config = app.config['DEEP_NEURAL_NETWORK_MODEL']
        model_name = 'DEEP_NEURAL_NETWORK_MODEL'

    elif model_type == 6:
        md = dss.DecisionTreeRegressor()
        model_config = app.config['DECISIONTREE_REGRESSOR_MODEL']
        model_name = 'DecisionTree_Regressor_MODEL'
    return md, model_config, model_name


class MachineLearning:
    def __init__(self, model_type, action_id, protection_goods_id, user_id=None):
        self.DSS, self.model_config, self.model_name = setDssNetwork(model_type)
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
        self.cache_key = str(self.action_id) + '_' + str(self.protection_goods_id)
        self.scaler_file_path = os.path.join(app.config['STORAGE_DIRECTORY'], 'scaler_' + str(self.action_id) + '_' +
                                             str(self.protection_goods_id) + '.save')

        self.trained_model_path = os.path.join(app.config['STORAGE_DIRECTORY'], str(self.action_id) + '_' +
                                               str(self.protection_goods_id) + '_' + self.model_name + '.sav')


    def process(self):

        # https://scikit-learn.org/stable/modules/tree.html

        if not path.exists(app.config['STORAGE_DIRECTORY']):
            os.mkdir(app.config['STORAGE_DIRECTORY'])
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
        # self.training_history_log()
        return {
            'status': 200,
            'message': 'Success'
        }

    def data_generation(self):

        # https://scikit-learn.org/stable/modules/tree.html

        if not path.exists(app.config['STORAGE_DIRECTORY']):
            os.mkdir(app.config['STORAGE_DIRECTORY'])
        self.data_load()
        min_max_scaler = preprocessing.MinMaxScaler()
        # Opening JSON file


        initial_obs = pd.read_csv("initial_observations.csv")
        initial_obs = pd.DataFrame(min_max_scaler.fit_transform(initial_obs), columns=initial_obs.columns,
                                 index=initial_obs.index)
        f = open('data_config.json', )
        config_file = json.load(f)
        main_Obs = {}
        column_names =[]
        feature_data = {}
        # if data is not None:
        #     for item in data[0]['features']:
        #         test =item['feature_name']
        #         column_names.append(str("alfa_"+item['feature_name']))

        # column_names = ["alfa_corrossion", "alfa_sediment_cover", "alfa_shipping_intensity","alfa_depth","alfa_tnt"]



        # alfa_df = pd.DataFrame(columns=column_names)
        # data_ = pd.DataFrame(columns=column_names)
        for index, row in initial_obs.iterrows():
            for key in initial_obs.columns:
                if key in config_file['features'].keys():
                    obj = config_file['features'][key]
                    alpha = None
                    if (obj['relation'] == 'inverse'):
                        alpha = row[key] * row.risk
                    else :
                        temp = row.risk
                        if temp == 0:
                            temp = 0.1
                            alpha = row[key] / temp
                        else:
                            if temp == 0:
                                temp = 0.1
                            alpha = row[key] / temp


                    config_file['features'][key]['alpha'].append(alpha)

        for key in initial_obs.columns:
            if key in config_file['features'].keys():
                alpha_avg = sum(config_file['features'][key]['alpha']) / len(config_file['features'][key]['alpha'])
                config_file['features'][key]['alpha_avg'] = alpha_avg

        # data_alfas = []
        # for row in initial_obs.itertuples(index=True, name='Pandas'):
        #     alfa_corrossion = row.corrosion * row.risk
        #     alfa_sediment_cover = row.sediment_cover * row.risk
        #     alfa_shipping_intensity = row.shipping_intensity / row.risk
        #     alfa_depth = row.depth * row.risk
        #     alfa_tnt = row.tnt / row.risk
        #     values = [alfa_corrossion, alfa_sediment_cover, alfa_shipping_intensity, alfa_depth, alfa_tnt]
        #     zipped = zip(column_names, values)
        #     a_dictionary = dict(zipped)
        #     print(a_dictionary)
        #     data_alfas.append(a_dictionary)
        #
        #     # print(row.corrosion, row.sediment_cover,row.shipping_intensity, row.depth,row.tnt, row.risk)
        # alfa_df = alfa_df.append(data_alfas, True)
        # counter_colnames = 0
        # alfa_calculations = {}
        # for col in column_names:
        #     alfa_calculations[str(col)] = alfa_df[col].sum(axis=0) / len(alfa_df)
        #     counter_colnames += 1
        # for label, row in initial_obs.iterrows():
        #     print(label)
        #     print(row)



        # returns JSON object as
        # a dictionary


        features_key = 'features'
        action_key = 'action_id'
        protection_key = 'protection_good_id'
        # Iterating through the json list
        observations = 5
        feature_data = {}
        main_data = {}
        directory = app.config['STORAGE_DIRECTORY']

        # if config_file is not None:
        #     counter = 0
        #     while counter < observations:
        #         feature_data[counter] = []
        #         for key in config_file['features']:
        #             feature_data[counter][key] = random.uniform(config_file['features'][key]['min_value'], config_file['features'][key]['max_value'])
        #             counter += 1

        # data_frame = pd.DataFrame.from_dict(feature_data, orient='index')
        # filename = str(config_file['action_id']) +'_'+ str(config_file['protection_good_id'])+"_generated_.csv"
        # file_path = os.path.join(directory, filename)
        # main_data[ str(config_file['action_id']) +'_'+ str(config_file['protection_good_id'])] = feature_data
        #
        # data_frame.to_csv(file_path, index=False)

        if config_file is not None:
            # for conf_item in config_file:
            counter = 0
            feature_data = {}
            while counter < observations:
                feature_data[counter] = {}
                for key in config_file['features'].keys():
                    item = config_file['features'].get(key)
                    feature_data[counter][key] =  random.uniform(item['min_value'], item['max_value'])
                counter += 1

            data_frame = pd.DataFrame.from_dict(feature_data, orient='index')

            data_frame = pd.DataFrame(min_max_scaler.fit_transform(data_frame.T), columns=data_frame.columns, index=data_frame.index)
            risk_key = 'risk'
            risk_final = []

            for row in data_frame.itertuples(index=True, name='Pandas'):
                temp_risk = []
                for key in config_file['features'].keys():
                    key_item = config_file['features'].get(key)
                    val = key_item['relation']
                    test_item = getattr(row, str(key))
                    weighted_sum = key_item['weight']
                    alpha = key_item['alpha_avg']
                    risk = 0
                    risk_scaled = 0
                    if (val == 'inverse'):
                        if test_item == 0:
                            test_item = 0.1
                        risk = (alpha / test_item)* (weighted_sum)
                        # risk_scaled = (risk / 100) * (weighted_sum)
                    else:
                        if test_item == 0:
                            test_item = 0.1
                        risk = (test_item / alpha)* (weighted_sum)
                        # risk_scaled = (risk / 100) * (weighted_sum)
                    temp_risk.append(risk)
                row_risk = sum(temp_risk)
                risk_final.append(row_risk)
            data_frame[risk_key] = risk_final

            # data_frame = pd.DataFrame(x_scaled)
            filename = str(config_file['action_id']) +'_'+ str(config_file['protection_good_id'])+"_generated_.csv"
            file_path = os.path.join(directory, filename)
            main_data[str(config_file['action_id']) +'_'+ str(config_file['protection_good_id'])] = feature_data

            data_frame.to_csv(file_path, index=False)
            f.close()
        # old file
        # feature_data = {}
        # if config_file is not None:
        #     counter = 0
        #     feature_data[counter] = {}
        #     while counter < observations:
        #
        #
        #
        #         for key in config_file['features'].keys():
        #             item = config_file['features'].get(key)
        #             # for item in config_file['features'].get(conf_item):
        #             feature_data[counter][key] =  random.uniform(item['min_value'], item['max_value'])
        #         counter += 1
        #     data_frame = pd.DataFrame.from_dict(feature_data, orient='index')
        #     filename = str(config_file['action_id']) +'_'+ str(config_file['protection_good_id'])+"_generated_.csv"
        #     file_path = os.path.join(di`rectory, filename)
        #     main_data[str(config_file['action_id']) +'_'+ str(config_file['protection_good_id'])] = feature_data
        #
        #     data_frame.to_csv(file_path, index=False)
        # f.close()
        # old file

        return {
            'status': 200,
            'message': 'Data Generated Successfully.'
        }

    def data_load(self):
        amucad = dt.Amucad()
        self.data, self.input_variables, self.output_variables = amucad.amucad_generic_api(self)

    def data_intialization(self):

        self.data.round(2)

        # drop nulls from response variable's columns
        for y in self.output_variables:
            self.data = self.data.dropna(how='any', subset=[y])

        # Separation of input variables
        # self.x_train = self.data
        # for y in self.output_variables:
        #     self.x_train = self.x_train.drop(y, axis=1)

        self.x_train = self.data[self.input_variables]

        # Separation of output variables
        self.y_train = self.data[self.output_variables]
        # for y in self.output_variables:
        #     self.y_train = self.data[y]
        #     break

    def data_preprocessing(self):

        # Numeric Imputation
        impute_numerical = SimpleImputer(strategy="mean")
        numerical_transformer = Pipeline(
            steps=[('impute_numerical', impute_numerical)])
        self.x_train = ColumnTransformer(transformers=[('numerical', numerical_transformer, self.input_variables)],
                                         remainder="passthrough").fit_transform(self.x_train)
        # from sklearn.datasets import make_regression
        # self.x_train = make_regression(n_samples=2000, n_features=10, n_informative=8, n_targets=2,
        #                                random_state=1)

        self.data_scaling()

    def data_scaling(self):
        # Data Scaling
        scaler = StandardScaler()
        # scaler.fit(self.x_train[0])
        scaler.fit(self.x_train)
        if os.path.exists(self.scaler_file_path):
            os.remove(self.scaler_file_path)
        joblib.dump(scaler, self.scaler_file_path)
        # self.x_train = scaler.transform(self.x_train[0])
        self.x_train = scaler.transform(self.x_train)

    def training(self):

        r = redis.Redis()
        r.delete(self.cache_key)
        r.mset({self.cache_key: json.dumps(self.output_variables)})

        return self.DSS.training(self)

    def set_test_data(self, data):
        self.test_data = [data]

    def testing(self):
        if os.path.exists(self.scaler_file_path) and os.path.exists(self.trained_model_path):
            # Before prediction
            self.apply_existing_scaler()
            return self.DSS.predict_data(self, self.test_data)
        else:
            return None

    def apply_existing_scaler(self):
        scaler = joblib.load(self.scaler_file_path)
        self.test_data = scaler.transform(self.test_data)

    def training_history_log(self):

        from pymongo import MongoClient
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
