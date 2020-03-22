from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from app.structure import dss
from app import app
import os
import pandas as pd

model_storage_folder = os.path.join(app.root_path, 'storage/models')
import numpy as np
from sklearn.impute import SimpleImputer


def setDssNetwork(model_type):
    md = model_config = model_name = None

    if (model_type == 1):
        md = dss.NeuralNetwork()
        model_config = app.config['NEURAL_NETWORK_MODEL']
        model_name = 'NEURAL_NETWORK_MODEL'

    elif (model_type == 2):
        md = dss.RandomForest()
        model_config = app.config['RANDOM_FOREST_CLASSIFIER_MODEL']
        model_name = 'RANDOM_FOREST_CLASSIFIER_MODEL'

    elif (model_type == 3):
        md = dss.LinearRegressionM()
        model_config = app.config['LINEAR_REGRESSION_MODEL']
        model_name = 'LINEAR_REGRESSION_MODEL'

    elif (model_type == 4):
        md = dss.LogisticRegressionM()
        model_config = app.config['LOGISTIC_REGRESSION_MODEL']
        model_name = 'LOGISTIC_REGRESSION_MODEL'

    return md, model_config, model_name


class Finding:
    def __init__(self, model_type, assesment_name):
        print(' Findings Constructor')
        # data members
        self.DSS, self.model_config, self.model_name = setDssNetwork(model_type)
        self.data = None
        self.x_train = None
        self.y_train = None
        self.trained_scaler_path = os.path.join(model_storage_folder,
                                                app.config['MODELS'][assesment_name] + self.model_config['scaler'])
        self.trained_model_path = os.path.join(model_storage_folder,
                                               app.config['MODELS'][assesment_name] + self.model_config['model'])
        if not os.path.exists(model_storage_folder):
            os.makedirs(model_storage_folder)

    def initiate_training(self, file):
        print(' Initiate Training Process')
        self.data_initialization(file)
        self.data_transformation()
        self.x_train = self.DSS.data_preprocessing(self)
        accuracy = self.DSS.training(self)
        print(accuracy)

        # To determine best model parameter
        # self.DSS.determineBestHyperParameters( self )

    def initiate_testing(self, file):
        print(' Initiate Testing Process')
        self.data_initialization(file)
        self.data_transformation()
        if os.path.exists(self.trained_scaler_path) and os.path.exists(self.trained_model_path):
            accuracy = self.DSS.testing(self)
            print(accuracy)
        else:
            print(' Model Not Found')

    def data_initialization(self, file):
        print(' Data Initialization ')

        self.data = pd.read_csv(file, usecols=self.features)
        self.data = self.data.dropna(how='any', subset=[self.response_variable])
        self.x_train = self.data.drop(self.response_variable, axis=1)  # axis 1 for column
        self.y_train = self.data[self.response_variable]

    def data_transformation(self):
        print(' Data Preprocessing ')

        labelEncoder_Y = LabelEncoder()
        self.y_train = labelEncoder_Y.fit_transform(self.y_train)

        # Numeric Imputation
        impute_numerical = SimpleImputer(strategy="mean")
        numerical_transformer = Pipeline(
            steps=[('impute_numerical', impute_numerical)])

        # Categorical Imputation and Hot Encoding
        impute_categorical = SimpleImputer(strategy="most_frequent")
        onehotencoder_categorical = OneHotEncoder(handle_unknown="ignore")
        categorical_transformer = Pipeline(
            steps=[('impute_categorical', impute_categorical),
                   ('onehotencoder_categorical', onehotencoder_categorical)])

        # Column Transformer
        self.x_train = ColumnTransformer(transformers=[('numerical', numerical_transformer, self.numericalColumns),
                                                       ('cat', categorical_transformer, self.categoricalColumns)],
                                         remainder="passthrough").fit_transform(self.x_train)

        print(self.x_train)


class Fish(Finding):

    def __init__(self, model_type, assesment_name):
        print(' Fish Constructor')
        super().__init__(model_type, assesment_name)


class FdiAssessment(Fish):

    def __init__(self, model_type):
        print(' FDI_ASSESMENT Constructor')
        super().__init__(model_type, app.config['MODELS_ID_MAPPING'][0])
        # All features
        self.features = [
            # "project",
            # "chemsea_sampleid",
            # "sample_id",
            "station",
            # "cruise",
            "year",
            "month",
            "day",
            # "fi_area",
            # "species",
            "group",
            "sex",
            "fish_no",
            "total_length",  # cm
            "total_weight",  # g
            "latitude",  # [dec deg]
            "longitude",  # [dec deg]
            "bottom_temperature",  # C
            "bottom_salinity",  # [PSU]
            "bottom_oxygen_saturation",  # [%]
            "hydrography_depth",  # [m]
            "fdi",  # fish disease index (FDI) (TI-FI)
            "fdi_assesment"
            # [G: green, good health status; Y: yellow, medium health status; R: red, bad health status]
        ]

        # Categorical Columns
        self.categoricalColumns = ["sex", "group"]

        # Numeric Columns
        self.numericalColumns = [
            "station",
            "year",
            "month",
            "day",
            "fish_no",
            "total_length",  # cm
            "total_weight",  # g
            "latitude",  # [dec deg]
            "longitude",  # [dec deg]
            "bottom_temperature",  # C
            "bottom_salinity",  # [PSU]
            "bottom_oxygen_saturation",  # [%]
            "hydrography_depth",  # [m]
            "fdi",  # fish disease index (FDI) (TI-FI)
        ]

        # Response Variable
        self.response_variable = 'fdi_assesment'

    def start(self):
        # training_file = 'wine_data.csv'
        training_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI.CSV'
        self.initiate_training(training_file)

        # testing_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI_TEST.CSV'
        # self.initiate_testing(testing_file)
