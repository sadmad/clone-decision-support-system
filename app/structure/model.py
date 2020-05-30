from flask import jsonify
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from app.structure import dss
from app import app
import os
import pandas as pd
import redis
import json

from sklearn.impute import SimpleImputer


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


class Finding:
    def __init__(self, model_type, assessment_name):

        print(' Findings Constructor')
        # data members
        self.DSS, self.model_config, self.model_name = setDssNetwork(model_type)
        self.data = None
        self.x_train = None
        self.y_train = None
        self.response_variable_key = "'" + self.model_name + '_' + app.config['MODELS'][
            assessment_name] + "response" + "'"
        self.assessment_name = assessment_name
        self.trained_scaler_path = os.path.join(app.config['STORAGE_DIRECTORY'],
                                                app.config['MODELS'][assessment_name] + self.model_config['scaler'])
        self.trained_model_path = os.path.join(app.config['STORAGE_DIRECTORY'],
                                               app.config['MODELS'][assessment_name] + self.model_config['model'])

    def initiate_training(self, file):
        print(' Initiate Training Process')
        self.data_initialization(file)
        self.data_transformation()
        self.x_train = self.DSS.data_preprocessing(self)
        accuracy = self.DSS.training(self)
        print(accuracy)
        return accuracy

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

    def predict_data(self, data):

        if os.path.exists(self.trained_scaler_path) and os.path.exists(self.trained_model_path):

            # Before prediction

            response = self.DSS.predict_data(self, data)
            print(response)
            return response
        else:
            print(' Model Not Found')
            return None

    def data_initialization(self, file):
        print(' Data Initialization ')

        self.data = pd.read_csv(file, usecols=self.features)
        self.data = self.data.dropna(how='any', subset=[self.response_variable])
        self.x_train = self.data.drop(self.response_variable,
                                      axis=1)  # axis 1 for column   self.data.iloc[:, :-1].values
        self.y_train = self.data[self.response_variable]  # self.data.iloc[:, -1].values

    def data_transformation(self):
        print(' Data Prepossessing ')

        """
        we can also use simple form for imputation and can define columns in fit 
         imputer.fit(X[:,1:3]) will impute from 1 to 2
        """
        if self.regression != 1:
            labelEncoder_Y = LabelEncoder()
            self.y_train = labelEncoder_Y.fit_transform(self.y_train)

            r = redis.Redis()
            r.delete(self.response_variable_key)
            r.mset({self.response_variable_key: json.dumps(labelEncoder_Y.classes_.tolist())})

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


class Munition(Finding):

    def __init__(self, model_type, assessment_name):
        print(' Munition Constructor')
        super().__init__(model_type, assessment_name)


class ExplosionFisheriesAssessment(Munition):

    def __init__(self, model_type):
        self.regression = 1  # 0 for classification
        print(' ExplosionFisheries ASSESSMENT Constructor')
        super().__init__(model_type, app.config['MODELS_ID_MAPPING'][10])
        # All features from data set file
        self.features = [
            'confidence_level',
            'coordinates_0',
            'coordinates_1',
            'ammunition_type_id',
            'ammunition_categories_id',
            'ammunition_sub_categories_id',
            'corrosion_level',  #
            'sediment_cover',
            'bio_cover',

            'traffic_intensity_shipping_fishing_2016_value',
            'physical_features_anoxic_level_probabilities_value',
            'biodiversity_benthic_habitats_bqr',
            'biodiversity_integrated_fish_assessments_bqr',
            'fisheries_fisheries_bottom_trawl_value',
            'fisheries_fisheries_surface_midwater_value',
            'fisheries_coastal_and_stationary_value',

            'Explosion_Fisheries',
        ]

        # Write only Categorical Columns here i.e strings or columns with ranges
        self.categoricalColumns = [
            'ammunition_type_id',
            'ammunition_categories_id',
            'ammunition_sub_categories_id',
        ]

        # Write only Numeric Columns here
        self.numericalColumns = [
            'confidence_level',
            'coordinates_0',
            'coordinates_1',
            'corrosion_level',  #
            'sediment_cover',
            'bio_cover',
            'traffic_intensity_shipping_fishing_2016_value',
            'physical_features_anoxic_level_probabilities_value',
            'biodiversity_benthic_habitats_bqr',
            'biodiversity_integrated_fish_assessments_bqr',
            'fisheries_fisheries_bottom_trawl_value',
            'fisheries_fisheries_surface_midwater_value',
            'fisheries_coastal_and_stationary_value',
        ]

        # Write only Response Variable: should be 1 variable
        self.response_variable = 'Explosion_Fisheries'

    def start(self):
        training_file = os.path.join(app.config['STORAGE_DIRECTORY'], "amucad_dataset.csv")
        return self.initiate_training(training_file)

    def getExplosionFisheriesAssessment(self, rs):
        col_13 = col_14 = col_15 = col_16 = col_17 = col_18 = col_19 = col_20 = col_21 = 0
        if rs['ammunition_type_id'] == 137:
            col_14 = 1
        if rs['ammunition_type_id'] == 138:
            col_15 = 1
        if rs['ammunition_type_id'] == 139:
            col_16 = 1
        if rs['ammunition_type_id'] == 140:
            col_17 = 1
        if rs['ammunition_type_id'] == 141:
            col_18 = 1
        if rs['ammunition_type_id'] == 85:
            col_13 = 1
        if rs['ammunition_categories_id'] == 1:
            col_19 = 1
        if rs['ammunition_sub_categories_id'] == 1:
            col_20 = 1
        if rs['ammunition_sub_categories_id'] == 2:
            col_21 = 1

        prediction = self.predict_data([[
            rs['confidence_level'], rs['coordinates_0'], rs['coordinates_1'],
            rs['corrosion_level'], rs['sediment_cover'], rs['bio_cover'],
            rs['traffic_intensity_shipping_fishing_2016_value'],
            rs['physical_features_anoxic_level_probabilities_value'],
            rs['biodiversity_benthic_habitats_bqr'],
            rs['biodiversity_integrated_fish_assessments_bqr'],
            rs['fisheries_fisheries_bottom_trawl_value'],
            rs['fisheries_fisheries_surface_midwater_value'],
            rs['fisheries_coastal_and_stationary_value'],
            col_13, col_14, col_15,
            col_16, col_17, col_18,
            col_19, col_20, col_21
        ]])
        prediction_number = status = message = 0
        if prediction is not None:
            prediction_number = json.loads(prediction)[0]
            message = ''
            status = 200
            if self.regression == 0:
                model_response_variable = json.loads(redis.Redis().get(self.response_variable_key))
                prediction_number = model_response_variable[prediction_number]

        else:
            message = 'Please Train Model First.'
            status = 422
        return {
            'prediction': prediction_number,
            'status': status,
            'message': message
        }


class ExplosionShippingAssessment(Munition):

    def __init__(self, model_type):
        self.regression = 1  # 0 for classification
        print(' ExplosionFisheries ASSESSMENT Constructor')
        super().__init__(model_type, app.config['MODELS_ID_MAPPING'][11])
        # All features from data set file
        self.features = [
            'confidence_level',
            'coordinates_0',
            'coordinates_1',
            'ammunition_type_id',
            'ammunition_categories_id',
            'ammunition_sub_categories_id',
            'corrosion_level',  #
            'sediment_cover',
            'bio_cover',

            'traffic_intensity_shipping_all_2016_value',
            'traffic_intensity_shipping_cargo_2016_value',
            'traffic_intensity_shipping_container_2016_value',
            'traffic_intensity_shipping_fishing_2016_value',
            'traffic_intensity_shipping_other_2016_value',
            'traffic_intensity_shipping_passenger_2016_value',
            'traffic_intensity_shipping_rorocargo_2016_value',
            'traffic_intensity_shipping_service_2016_value',
            'traffic_intensity_shipping_tanker_2016_value',
            'physical_features_seabed_slope_value',
            'bathymetry_depth_value',

            'Explosion_Shipping',
        ]

        # Write only Categorical Columns here i.e strings or columns with ranges
        self.categoricalColumns = [
            'ammunition_type_id',
            'ammunition_categories_id',
            'ammunition_sub_categories_id',
        ]

        # Write only Numeric Columns here
        self.numericalColumns = [
            'confidence_level',
            'coordinates_0',
            'coordinates_1',
            'corrosion_level',  #
            'sediment_cover',
            'bio_cover',
            'traffic_intensity_shipping_all_2016_value',
            'traffic_intensity_shipping_cargo_2016_value',
            'traffic_intensity_shipping_container_2016_value',
            'traffic_intensity_shipping_fishing_2016_value',
            'traffic_intensity_shipping_other_2016_value',
            'traffic_intensity_shipping_passenger_2016_value',
            'traffic_intensity_shipping_rorocargo_2016_value',
            'traffic_intensity_shipping_service_2016_value',
            'traffic_intensity_shipping_tanker_2016_value',
            'physical_features_seabed_slope_value',
            'bathymetry_depth_value',
        ]

        # Write only Response Variable: should be 1 variable
        self.response_variable = 'Explosion_Shipping'

    def start(self):
        training_file = os.path.join(app.config['STORAGE_DIRECTORY'], "amucad_dataset.csv")
        return self.initiate_training(training_file)

    def getExplosionShippingAssessment(self, rs):
        col_13 = col_14 = col_15 = col_16 = col_17 = col_18 = col_19 = col_20 = col_21 = 0

        prediction = self.predict_data([[
            rs['confidence_level'], rs['coordinates_0'], rs['coordinates_1'],
            rs['corrosion_level'], rs['sediment_cover'], rs['bio_cover'],
            rs['traffic_intensity_shipping_all_2016_value'],
            rs['traffic_intensity_shipping_cargo_2016_value'],
            rs['traffic_intensity_shipping_container_2016_value'],
            rs['traffic_intensity_shipping_fishing_2016_value'],
            rs['traffic_intensity_shipping_other_2016_value'],
            rs['traffic_intensity_shipping_passenger_2016_value'],
            rs['traffic_intensity_shipping_rorocargo_2016_value'],
            rs['traffic_intensity_shipping_service_2016_value'],
            rs['traffic_intensity_shipping_tanker_2016_value'],
            rs['physical_features_seabed_slope_value'],
            rs['bathymetry_depth_value'],
            col_13, col_14, col_15,
            col_16, col_17, col_18,
            col_19, col_20, col_21
        ]])
        prediction_number = status = message = 0
        if prediction is not None:
            prediction_number = json.loads(prediction)[0]
            message = ''
            status = 200
            if self.regression == 0:
                model_response_variable = json.loads(redis.Redis().get(self.response_variable_key))
                prediction_number = model_response_variable[prediction_number]

        else:
            message = 'Please Train Model First.'
            status = 422
        return {
            'prediction': prediction_number,
            'status': status,
            'message': message
        }


class Fish(Finding):

    def __init__(self, model_type, assessment_name):
        print(' Fish Constructor')
        super().__init__(model_type, assessment_name)


class FdiAssessment(Fish):

    def __init__(self, model_type):
        self.regression = 0
        print(' FDI_ASSESMENT Constructor')
        super().__init__(model_type, app.config['MODELS_ID_MAPPING'][0])
        # All features from data set file
        self.features = [
            # "project",
            # "chemsea_sampleid",
            # "sample_id",
            # "cruise",
            # "fi_area",
            # "species",
            "station",
            "year",
            "month",
            "day",
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

        # Write only Categorical Columns here i.e strings or columns with ranges
        self.categoricalColumns = ["sex", "group"]

        # Write only Numeric Columns here
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

        # Write only Response Variable: should be 1 variable
        self.response_variable = 'fdi_assesment'

    def start(self):
        training_file = os.path.join(app.config['STORAGE_DIRECTORY'], "DAIMON_Cod_Data_FDI.CSV")
        return self.initiate_training(training_file)
        # testing_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI_TEST.CSV'
        # self.initiate_testing(testing_file)

    def getFdiAssessment(self, data):
        sex_m = sex_n = sex_w = 0
        if data['sex'] == 'm':
            sex_m = 1
        elif data['sex'] == 'n':
            sex_n = 1
        elif data['sex'] == 'w':
            sex_w = 1

        group_EXT = group_LEEXT = 0
        if data['group'] == 'EXT':
            group_EXT = 1
        if data['group'] == 'LEEXT':
            group_LEEXT = 1

        prediction = self.predict_data([[data['station'], data['year'], data['month'], data['day'], data['fish_no'],
                                         data['total_length'], data['total_weight'], data['latitude'],
                                         data['longitude'],
                                         data['bottom_temperature'],
                                         data['bottom_salinity'], data['bottom_oxygen_saturation'],
                                         data['hydrography_depth'],
                                         data['fdi'],
                                         sex_m, sex_n, sex_w, group_EXT, group_LEEXT]])

        prediction_number = status = message = 0
        if prediction is not None:

            prediction_number = json.loads(prediction)[0]
            prd_response = prediction_number
            if self.regression == 0:
                model_response_variable = json.loads(redis.Redis().get(self.response_variable_key))
                prd_response = model_response_variable[prediction_number]
            status = 200
            message = 'success'

            assessment_name = self.assessment_name
            model_name = self.model_name
            prediction_response = prd_response

        else:
            message = 'Please train model first.'
            status = 422

        message = {
            'status': status,
            'data': {
                'assessment': assessment_name,
                'model_name': model_name,
                'prediction': prediction_response,
                'message': message
            },
        }
        resp = jsonify(message)
        resp.status_code = status
        return resp


class CFAssessment(Fish):

    def __init__(self, model_type):
        self.regression = 0
        print(' FDI_ASSESMENT Constructor')
        super().__init__(model_type, app.config['MODELS_ID_MAPPING'][1])
        # All features from data set file
        self.features = [
            'Cryp1',
            'Cryp2',
            'Cryp3',
            'EpPap1',
            'EpPap2',
            'EpPap3',
            'FinRot',  #
            'Locera1',
            'Locera2',
            'Locera3',
            'PBT',  #
            'Skel1',
            'Skel2',
            'Skel3',
            'Ulc1',
            'Ulc2',
            'Ulc3',
            'condition_factor',  #
            'cf_assessment'
        ]

        # Write only Categorical Columns here i.e strings or columns with ranges
        self.categoricalColumns = [

        ]

        # Write only Numeric Columns here
        self.numericalColumns = [
            'Cryp1',
            'Cryp2',
            'Cryp3',
            'EpPap1',
            'EpPap2',
            'EpPap3',
            'FinRot',
            'Locera1',
            'Locera2',
            'Locera3',
            'PBT',
            'Skel1',
            'Skel2',
            'Skel3',
            'Ulc1',
            'Ulc2',
            'Ulc3',
            'condition_factor'
        ]

        # Write only Response Variable: should be 1 variable
        self.response_variable = 'cf_assessment'

    def start(self):
        training_file = os.path.join(app.config['STORAGE_DIRECTORY'], "DAIMON_Cod_Data_FDI.CSV")
        return self.initiate_training(training_file)

        # testing_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI_TEST.CSV'
        # self.initiate_testing(testing_file)

    def getCFAssessment(self,data):
        prediction = self.predict_data(
            [[data['Cryp1'], data['Cryp2'], data['Cryp3'], data['EpPap1'], data['EpPap2'], data['EpPap3'],
              data['FinRot'], data['Locera1'], data['Locera2'], data['Locera3'], data['PBT'], data['Skel1'],
              data['Skel2'], data['Skel3'], data['Ulc1'], data['Ulc2'], data['Ulc3'], data['condition_factor']]])
        prediction_number = status = message = 0
        if prediction is not None:
            prediction_number = json.loads(prediction)[0]
            prd_response = prediction_number
            if self.regression == 0:
                model_response_variable = json.loads(redis.Redis().get(self.response_variable_key))
                prd_response = model_response_variable[prediction_number]
            status = 200
            message = 'success'

            assessment_name = self.assessment_name
            model_name = self.model_name
            prediction_response = prd_response

        else:
            message = 'Please train model first.'
            status = 422

        message = {
            'status': status,
            'data': {
                'assessment': assessment_name,
                'model_name': model_name,
                'prediction': prediction_response,
                'message': message
            },
        }
        resp = jsonify(message)
        resp.status_code = status
        return resp