import os

from app.structure import data_transformer as dt, dss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from app import app
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib


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

    return md, model_config, model_name


class MachineLearning:
    def __init__(self, model_type, action_id, protection_goods_id):
        self.DSS, self.model_config, self.model_name = setDssNetwork(model_type)
        self.action_id = action_id
        self.protection_goods_id = protection_goods_id
        self.data = None
        self.input_variables = None
        self.output_variables = None
        self.x_train = None
        self.y_train = None
        self.regression = None
        self.scaler_file_path = os.path.join(app.config['STORAGE_DIRECTORY'], 'scaler_' + str(self.action_id) + '_' +
                                             str(self.protection_goods_id) + '.save')

        self.trained_model_path = os.path.join(app.config['STORAGE_DIRECTORY'], str(self.action_id) + '_' +
                                               str(self.protection_goods_id) + '_' + self.model_name + '.sav')

    def process(self):
        self.data_intialization()
        self.data_preprocessing()
        self.training()

        data = [[
            340,
            0.6,
            0.37,
            3.8,
            290,
            8.12,
            390,
            0
        ]]
        response = self.testing(data)
        print(response)


    def data_intialization(self):
        amucad = dt.Amucad()
        self.data, self.input_variables, self.output_variables = amucad.amucad_generic_api(self)

        # drop nulls from response variable's columns
        for y in self.output_variables:
            self.data = self.data.dropna(how='any', subset=[y])

        # Separation of input variables
        self.x_train = self.data
        for y in self.output_variables:
            self.x_train = self.x_train.drop(y, axis=1)

        # Separation of output variables
        self.y_train = self.data[self.output_variables]

    def data_preprocessing(self):

        # Numeric Imputation
        impute_numerical = SimpleImputer(strategy="mean")
        numerical_transformer = Pipeline(
            steps=[('impute_numerical', impute_numerical)])
        self.x_train = ColumnTransformer(transformers=[('numerical', numerical_transformer, self.input_variables)],
                                         remainder="passthrough").fit_transform(self.x_train)

        self.data_scaling()

    def data_scaling(self):

        # Data Scaling
        scaler = StandardScaler()
        scaler.fit(self.x_train)
        # https://stackoverflow.com/questions/41993565/save-minmaxscaler-model-in-sklearn
        if os.path.exists(self.scaler_file_path):
            os.remove(self.scaler_file_path)
        joblib.dump(scaler, self.scaler_file_path)
        self.x_train = scaler.transform(self.x_train)

    def training(self):
        accuracy = self.DSS.training(self)
        pass

    def testing(self, data):

        if os.path.exists(self.scaler_file_path) and os.path.exists(self.trained_model_path):
            # Before prediction
            response = self.DSS.predict_data(self, data)
            print(response)
            return response
        else:
            print(' Model Not Found')
            return None
