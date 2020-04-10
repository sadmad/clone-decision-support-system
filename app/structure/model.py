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

model_storage_folder = os.path.join(app.root_path, 'storage/models')

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

    return md, model_config, model_name


class Finding:
    def __init__(self, model_type, assesment_name):
        print(' Findings Constructor')
        # data members
        self.DSS, self.model_config, self.model_name = setDssNetwork(model_type)
        self.data = None
        self.x_train = None
        self.y_train = None
        self.response_variable_key = "'" + self.model_name + '_' + app.config['MODELS'][
            assesment_name] + "response" + "'"
        self.assessment_name = assesment_name
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


        # remove features with low variance 80% no change in below example
        # from sklearn.feature_selection import VarianceThreshold
        # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        # test1 = sel.fit_transform(self.x_train)
        # print('awais')


        from numpy import set_printoptions
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_classif

        test = SelectKBest(score_func=f_classif, k=4)
        fit = test.fit(self.x_train, self.y_train)
        # summarize scores
        set_printoptions(precision=3)
        print(fit.scores_)
        features = fit.transform(self.x_train)
        # summarize selected features
        print(features)



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
            response = self.DSS.predict_data(self, data)
            print(response)
            return response
        else:
            print(' Model Not Found')
            return None

    def data_initialization(self, file):
        print(' Data Initialization ')

        self.data = pd.read_csv(file, usecols=self.features)

        count_total_nan = self.data.isna().sum()
        count_total_not_nan = self.data.isna().sum()
        print(count_total_nan)
        print(count_total_not_nan)
        self.data = self.data.dropna(how='any', subset=[self.response_variable])
        self.x_train = self.data.drop(self.response_variable, axis=1)  # axis 1 for column
        self.y_train = self.data[self.response_variable]

    def data_transformation(self):
        print(' Data Prepossessing ')

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
        return self.initiate_training(training_file)

        # testing_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI_TEST.CSV'
        # self.initiate_testing(testing_file)


########

class CFAssessment(Fish):

    def __init__(self, model_type):
        print(' FDI_ASSESMENT Constructor')
        super().__init__(model_type, app.config['MODELS_ID_MAPPING'][1])
        # All features
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

        # Categorical Columns
        self.categoricalColumns = [

        ]

        # Numeric Columns
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

        # Response Variable
        self.response_variable = 'cf_assessment'

    def start(self):
        # training_file = 'wine_data.csv'
        training_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI.CSV'
        return self.initiate_training(training_file)

        # testing_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI_TEST.CSV'
        # self.initiate_testing(testing_file)


class LHIAssessment(Fish):

    def __init__(self, model_type):
        print(' LHI_ASSESMENT Constructor')
        super().__init__(model_type, app.config['MODELS_ID_MAPPING'][2])
        # All features
        self.features = [
            'liver_histo_index',
            'lhi_assessment'
        ]

        # Categorical Columns
        self.categoricalColumns = [

        ]

        # Numeric Columns
        self.numericalColumns = [
            'liver_histo_index'
        ]

        # Response Variable
        self.response_variable = 'lhi_assessment'

    def start(self):
        # training_file = 'wine_data.csv'
        training_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI.CSV'
        return self.initiate_training(training_file)

        # testing_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI_TEST.CSV'
        # self.initiate_testing(testing_file)


class MuscelCWAAssessment(Fish):

    def __init__(self, model_type):
        print(' MuscelCWA_ASSESMENT Constructor')
        super().__init__(model_type, app.config['MODELS_ID_MAPPING'][3])
        # All features
        self.features = [
            'Headkidney_LMS_Peak_1',
            'Headkidney_LMS_Peak_2',
            'Headkidney_lipofuscin',
            'normGST',
            'normCAT',
            'normGR',
            'normAChE',
            'muscle_cryo_sample_ID',
            'liver_cryo_sample_ID',
            'muscle_CWA_sample_ID',
            'muscelCWA_assessment'
        ]

        # Categorical Columns
        self.categoricalColumns = [

        ]

        # Numeric Columns
        self.numericalColumns = [
            'Headkidney_LMS_Peak_1',
            'Headkidney_LMS_Peak_2',
            'Headkidney_lipofuscin',
            'normGST',
            'normCAT',
            'normGR',
            'normAChE',
            'muscle_cryo_sample_ID',
            'liver_cryo_sample_ID',
            'muscle_CWA_sample_ID',
        ]

        # Response Variable
        self.response_variable = 'muscelCWA_assessment'

    def start(self):
        # training_file = 'wine_data.csv'
        training_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI.CSV'
        return self.initiate_training(training_file)

        # testing_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI_TEST.CSV'
        # self.initiate_testing(testing_file)


class LiverCWAAssessment(Fish):

    def __init__(self, model_type):
        print(' LiverCWA_ASSESMENT Constructor')
        super().__init__(model_type, app.config['MODELS_ID_MAPPING'][4])
        # All features
        self.features = [
            'bile_CWA_sample_ID',
            'liver_CWA_sample_ID',
            'liver_CWA_assessment'
        ]

        # Categorical Columns
        self.categoricalColumns = [
        ]

        # Numeric Columns
        self.numericalColumns = [
            'bile_CWA_sample_ID',
            'liver_CWA_sample_ID',
        ]

        # Response Variable
        self.response_variable = 'liver_CWA_assessment'

    def start(self):
        # training_file = 'wine_data.csv'
        training_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI.CSV'
        return self.initiate_training(training_file)

        # testing_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI_TEST.CSV'
        # self.initiate_testing(testing_file)


class ERYAssessment(Fish):

    def __init__(self, model_type):
        print(' ERY_ASSESSMENT Constructor')
        super().__init__(model_type, app.config['MODELS_ID_MAPPING'][5])
        # All features
        self.features = [
            'urine_CWA_sample_ID',
            'blood_1_sample_ID',
            'Erythrocytes',
            'ERY_assessment'
        ]

        # Categorical Columns
        self.categoricalColumns = [

        ]

        # Numeric Columns
        self.numericalColumns = [
            'urine_CWA_sample_ID',
            'blood_1_sample_ID',
            'Erythrocytes'
        ]

        # Response Variable
        self.response_variable = 'ERY_assessment'

    def start(self):
        # training_file = 'wine_data.csv'
        training_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI.CSV'
        return self.initiate_training(training_file)

        # testing_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI_TEST.CSV'
        # self.initiate_testing(testing_file)


class HBAssessment(Fish):

    def __init__(self, model_type):
        print(' HB_ASSESSMENT Constructor')
        super().__init__(model_type, app.config['MODELS_ID_MAPPING'][6])
        # All features
        self.features = [
            'Hemoglobin',
            'HB_assessment'
        ]

        # Categorical Columns
        self.categoricalColumns = [

        ]

        # Numeric Columns
        self.numericalColumns = [
            'Hemoglobin'
        ]

        # Response Variable
        self.response_variable = 'HB_assessment'

    def start(self):
        # training_file = 'wine_data.csv'
        training_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI.CSV'
        return self.initiate_training(training_file)

        # testing_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI_TEST.CSV'
        # self.initiate_testing(testing_file)


class GLUAssessment(Fish):

    def __init__(self, model_type):
        print(' GLU_ASSESSMENT Constructor')
        super().__init__(model_type, app.config['MODELS_ID_MAPPING'][7])
        # All features
        self.features = [
            'Glucose',
            'GLU_Assessment'
        ]

        # Categorical Columns
        self.categoricalColumns = [

        ]

        # Numeric Columns
        self.numericalColumns = [
            'Glucose'
        ]

        # Response Variable
        self.response_variable = 'GLU_Assessment'

    def start(self):
        # training_file = 'wine_data.csv'
        training_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI.CSV'
        return self.initiate_training(training_file)

        # testing_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI_TEST.CSV'
        # self.initiate_testing(testing_file)


class HCTAssessment(Fish):

    def __init__(self, model_type):
        print(' HCT_ASSESSMENT Constructor')
        super().__init__(model_type, app.config['MODELS_ID_MAPPING'][8])
        # All features
        self.features = [
            'Hematocrit',
            'HCT_Assessment'
        ]

        # Categorical Columns
        self.categoricalColumns = [

        ]

        # Numeric Columns
        self.numericalColumns = [
            'Hematocrit'
        ]

        # Response Variable
        self.response_variable = 'HCT_Assessment'

    def start(self):
        # training_file = 'wine_data.csv'
        training_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI.CSV'
        return self.initiate_training(training_file)

        # testing_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI_TEST.CSV'
        # self.initiate_testing(testing_file)


class GillCWAAssessment(Fish):

    def __init__(self, model_type):
        print(' GillCWA_ASSESSMENT Constructor')
        super().__init__(model_type, app.config['MODELS_ID_MAPPING'][9])
        # All features
        self.features = [
            'blood_2_sample_ID',
            'plasma_sample_ID',
            'liver_cryo_sample_ID',
            'head_kidney_cryo_sample_ID',
            'kidney_cryo_sample_ID',
            'liver_histo_routine_sample_ID',
            'histo_liver_tumour_sample_ID',
            'Histo_sample_ID',
            'gill_CWA_sample_ID',
            'gill_CWA_Assessment ',
            'otoliths_sample_ID'
        ]

        # Categorical Columns
        self.categoricalColumns = [

        ]

        # Numeric Columns
        self.numericalColumns = [
            'blood_2_sample_ID',
            'plasma_sample_ID',
            'liver_cryo_sample_ID',
            'head_kidney_cryo_sample_ID',
            'kidney_cryo_sample_ID',
            'liver_histo_routine_sample_ID',
            'histo_liver_tumour_sample_ID',
            'Histo_sample_ID',
            'gill_CWA_sample_ID'
        ]

        # Response Variable
        self.response_variable = 'gill_CWA_Assessment'

    def start(self):
        # training_file = 'wine_data.csv'
        training_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI.CSV'
        return self.initiate_training(training_file)

        # testing_file = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data_FDI_TEST.CSV'
        # self.initiate_testing(testing_file)
