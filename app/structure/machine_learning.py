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

    elif model_type == 6:
        md = dss.DecisionTreeRegressor()
        model_config = app.config['DECISIONTREE_REGRESSOR_MODEL']
        model_name = 'DecisionTree_Regressor_MODEL'
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
        self.is_regression = 1
        self.test_data = None
        self.scaler_file_path = os.path.join(app.config['STORAGE_DIRECTORY'], 'scaler_' + str(self.action_id) + '_' +
                                             str(self.protection_goods_id) + '.save')

        self.trained_model_path = os.path.join(app.config['STORAGE_DIRECTORY'], str(self.action_id) + '_' +
                                               str(self.protection_goods_id) + '_' + self.model_name + '.sav')

    def process(self):

        # https://scikit-learn.org/stable/modules/tree.html
        self.data_intialization()

        self.data_preprocessing()

        # from sklearn.datasets import make_regression
        # self.x_train, self.y_train = make_regression(n_samples=2000, n_features=10, n_informative=8, n_targets=2, random_state=1)

        accuracy = self.training()

        return self.testing()

    def data_intialization(self):

        amucad = dt.Amucad()
        self.data, self.input_variables, self.output_variables = amucad.amucad_generic_api(self)

        self.data.round(2)

        # drop nulls from response variable's columns
        for y in self.output_variables:
            self.data = self.data.dropna(how='any', subset=[y])

        # Separation of input variables
        self.x_train = self.data
        for y in self.output_variables:
            self.x_train = self.x_train.drop(y, axis=1)

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
        return self.DSS.training(self)

    def testing(self):

        self.test_data = [[
            340,
            0.6,
            0.37,
            3.8,
            290,
            8.12,
            390,
            0
        ]]

        self.apply_existing_scaler()
        if os.path.exists(self.scaler_file_path) and os.path.exists(self.trained_model_path):
            # Before prediction
            return self.DSS.predict_data(self, self.test_data)
        else:
            print(' Model Not Found')
            return None

    def apply_existing_scaler(self):
        scaler = joblib.load(self.scaler_file_path)
        self.test_data = scaler.transform(self.test_data)
