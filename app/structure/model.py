
from app.structure import dss
from app import app
import os
import pandas as pd


model_storage_folder = os.path.join(app.root_path, 'storage/models')
def setDssNetwork( model_type ):
    
    md = model_config = model_name = None

    if (model_type == 1):
        md = dss.NeuralNetwork()
        model_config  = app.config['NEURAL_NETWORK_MODEL']
        model_name    = 'NEURAL_NETWORK_MODEL'
    
    elif (model_type == 2):
        md = dss.RandomForest()
        model_config  = app.config['RANDOM_FOREST_CLASSIFIER_MODEL']
        model_name    = 'RANDOM_FOREST_CLASSIFIER_MODEL'
    
    elif (model_type == 3):
        md = dss.LinearRegressionM()
        model_config  = app.config['LINEAR_REGRESSION_MODEL']
        model_name    = 'LINEAR_REGRESSION_MODEL'
    
    elif (model_type == 4):
        md = dss.LogisticRegressionM()
        model_config  = app.config['LOGISTIC_REGRESSION_MODEL']
        model_name    = 'LOGISTIC_REGRESSION_MODEL'
    
    return md, model_config, model_name












class Finding:
    def __init__(self, model_type, assesment_name ):
        print(' Findings Constructor')
        #data members
        self.DSS, self.model_config, self.model_name = setDssNetwork(model_type)
        self.data                   = None
        self.x_train                = None
        self.y_train                = None
        self.trained_scaler_path    = os.path.join( model_storage_folder, app.config['MODELS'][assesment_name] + self.model_config['scaler'] )
        self.trained_model_path     = os.path.join( model_storage_folder, app.config['MODELS'][assesment_name] + self.model_config['model'] )
        if not os.path.exists(model_storage_folder):
            os.makedirs(model_storage_folder)

    def intiate_training( self, file ):
        print(' Intiate Training Process')
        self.data_initialization( file )
        self.categorical_fields_handling()
        self.x_train = self.DSS.data_preprocessing( self )
        accuracy = self.DSS.training( self )
        print(accuracy)
        # To determine best model parameter
        # self.DSS.determineBestHyperParameters( self )

    def intiate_testing( self, file ):
        print(' Intiate Testing Process')
        self.data_initialization( file )
        self.categorical_fields_handling()
        #self.x_train = self.DSS.data_preprocessing( self )
        if os.path.exists(self.trained_scaler_path) and os.path.exists(self.trained_model_path) :
            accuracy     = self.DSS.testing( self )
            print(accuracy)
        else:
            print(' Model Not Found')
        

class Fish(Finding):

    def __init__(self, model_type, assesment_name ):

        print(' Fish Constructor')
        super().__init__( model_type, assesment_name )

        #Data Members
        self.features = [
            "Cultivator",
            "Alchol",
            "Malic_Acid",
            "Ash",
            "Alcalinity_of_Ash",
            "Magnesium",
            "Total_phenols",
            "Falvanoids",
            "Nonflavanoid_phenols",
            "Proanthocyanins",
            "Color_intensity",
            "Hue",
            "OD280",
            "Proline"
        ]




class FDI_ASSESMENT(Fish):
    
    def __init__(self, model_type):

        print(' FDI_ASSESMENT Constructor')
        super().__init__( model_type, app.config['MODELS_ID_MAPPING'][0] )

    
    def categorical_fields_handling( self ):
        
        print(' Categorical Fields Handling FDI_ASSESMENT')
        #print(self.data)
        pass

    def data_initialization( self, file ):
        
        print(' Data Initialization FDI_ASSESMENT')
        self.data    = pd.read_csv( file, names=self.features )
        self.x_train = self.data.drop('Cultivator', axis=1)
        self.y_train = self.data['Cultivator']
        
    def start(self):
        
        training_file = 'wine_data.csv'
        self.intiate_training( training_file )
        
        testing_file  = 'wine_data_test.csv'
        self.intiate_testing( testing_file )

        
        



