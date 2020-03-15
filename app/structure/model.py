from app.structure import dss
from app import app
import os
import pandas as pd

model_storage_folder = os.path.join(app.root_path, 'storage/models')


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

        # Waleed Part
        self.categorical_fields_handling()
        # self.x_train = self.DSS.data_preprocessing( self )
        # accuracy = self.DSS.training( self )
        # print(accuracy)

        # To determine best model parameter
        # self.DSS.determineBestHyperParameters( self )

    def initiate_testing(self, file):
        print(' Initiate Testing Process')
        self.data_initialization(file)
        self.categorical_fields_handling()
        if os.path.exists(self.trained_scaler_path) and os.path.exists(self.trained_model_path):
            accuracy = self.DSS.testing(self)
            print(accuracy)
        else:
            print(' Model Not Found')


class Fish(Finding):

    def __init__(self, model_type, assesment_name):
        print(' Fish Constructor')
        super().__init__(model_type, assesment_name)

        # Data Members


class FDI_ASSESMENT(Fish):

    def __init__(self, model_type):
        print(' FDI_ASSESMENT Constructor')
        super().__init__(model_type, app.config['MODELS_ID_MAPPING'][0])
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
            # "group",
            "fish_no",
            # "sex",
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

    def data_initialization(self, file):
        print(' Data Initialization FDI_ASSESMENT')
        self.data = pd.read_csv(file, usecols = self.features)
        
        #print(self.features)
        print(self.data)
        
        # for wine dataset
        # self.x_train = self.data.drop('Cultivator', axis=1)
        # self.y_train = self.data['Cultivator']

        # For Fish data
        self.x_train = self.data.drop('fdi_assesment', axis=1)
   
       
      
        self.y_train = self.data['fdi_assesment']

        # from sklearn.preprocessing import Imputer
        # imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        # imputer = imputer.fit(self.x_train)

        # self.x_train = imputer.transform(self.x_train)
    
        print(self.x_train.round(decimals=2))
        
        #https://towardsdatascience.com/preprocessing-regression-imputation-of-missing-continuous-values-f612179bafb4


        #import pandas as pd
        import numpy as np
        # explicitly require this experimental feature
        from sklearn.experimental import enable_iterative_imputer
        # now you can import normally from sklearn.impute
        from sklearn.impute import IterativeImputer
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.linear_model import BayesianRidge
        import random


        imputer = IterativeImputer(BayesianRidge())
        #impute_data = pd.DataFrame(imputer.fit_transform(self.x_train))

        print(imputer.fit_transform(self.x_train))
        
        # import numpy as np
        # from sklearn.experimental import enable_iterative_imputer
        # from sklearn.impute import IterativeImputer
        # imp = IterativeImputer(max_iter=10, random_state=0)
        # imp.fit([[1, 2], [3, 6], [4, 8], [np.nan, 3], [7, np.nan]])

        # X_test = [[np.nan, 2], [6, np.nan], [np.nan, 6]]
        # # the model learns that the second feature is double the first
        # print(np.round(imp.transform(X_test)))



    def categorical_fields_handling(self):
        print(' Categorical Fields Handling FDI_ASSESMENT')

        pass

    def start(self):
        # training_file = 'wine_data.csv'
        training_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI.CSV'
        self.initiate_training(training_file)

        # testing_file  = 'wine_data_test.csv'
        # self.initiate_testing( testing_file )
