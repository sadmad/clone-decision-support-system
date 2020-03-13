
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
        
        #Waleed Part
        self.categorical_fields_handling()
        
        print(self.x_train)

        #self.x_train = self.DSS.data_preprocessing( self )
        #accuracy = self.DSS.training( self )
        #print(accuracy)
        

        # To determine best model parameter
        # self.DSS.determineBestHyperParameters( self )

    def intiate_testing( self, file ):
        print(' Intiate Testing Process')
        self.data_initialization( file )
        self.categorical_fields_handling()
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
            "Project",
            "CHEMSEA Sample ID",
            "sampleID",
            "station",
            "cruise",
            "year",
            "month",
            "day",
            "FI area",
            "species",
            "group",
            "fish no.",
            "sex",
            "total length [cm]",
            "total weight [g]",
            "latitude [dec deg]",
            "longitude [dec deg]",
            "bottom temperature [C]",
            "bottom salinity [PSU]",
            "bottom oxygen saturation [%]",
            "hydrography depth [m]",
            "fish disease index (FDI) (TI-FI)",
            "FDI Assessment (TI-FI) [G: green, good health status; Y: yellow, medium health status; R: red, bad health status]",
            "Cryp1",
            "Cryp2",
            "Cryp3",
            "EpPap1",
            "EpPap2",
            "EpPap3",
            "FinRot",
            "Locera1",
            "Locera2",
            "Locera3",
            "PBT",
            "Skel1",
            "Skel2",
            "Skel3",
            "Ulc1",
            "Ulc2",
            "Ulc3",
            "condition factor(CF) [total weight *100/total length^3] (TI-FI)",
            "CF Assessment (TI-FI) [G: green, good health status; Y: yellow, medium health status; R: red, bad health status]",
            "liver histo index (LHI) (TI-FI)",
            "LHI Assessment (TI-FI) [G: green, good health status; Y: yellow, medium health status; R: red, bad health status]",
            "Headkidney LMS Peak 1 (min) (AWI)",
            "Headkidney LMS Peak 2 (min) (AWI)",
            "Headkidney lipofuscin (area%) (AWI)",
            "normGST (SYKE)",
            "normCAT (SYKE)",
            "normGR (SYKE)",
            "normAChE (SYKE)",
            "muscle cryo (SYKE) sample ID",
            "liver cryo (SYKE) sample ID",
            "muscle CWA (VERIFIN) sample ID",
            "muscel CWA Assessment (VERIFIN) [G: green, no contamination; R: red, contamination]",
            "bile CWA (VERIFIN) sample ID",
            "liver CWA (VERIFIN) sample ID",
            "liver CWA Assessment (VERIFIN) [G: green, no contamination; R: red, contamination]",
            "urine CWA (VERIFIN) sample ID",
            "blood 1 (TI-FI) sample ID",
            "Erythrocytes [Mio/mL]",
            "ERY Assessment (TI-FI) [G: green, good health status; Y: yellow, medium health status; R: red, bad health status]",
            "Hemoglobin [mg/dL]",
            "HB Assessment (TI-FI) [G: green, good health status; Y: yellow, medium health status; R: red, bad health status]",
            "Glucose [mmol/L]",
            "GLU Assessment (TI-FI) [G: green, good health status; Y: yellow, medium health status; R: red, bad health status]",
            "Hematocrit [%]",
            "HCT Assessment (TI-FI) [G: green, good health status; Y: yellow, medium health status; R: red, bad health status]",
            "blood 2 (TI-FI) sample ID",
            "plasma (TI-FI) sample ID",
            "liver cryo (TI-FI/AWI) sample ID",
            "head kidney cryo (AWI) sample ID",
            "kidney cryo (AWI) sample ID",
            "liver histo routine (TI-FI) sample ID",
            "histo liver tumour (TI-FI) sample ID",
            "Histo (AWI) sample ID",
            "gill CWA (VERIFIN) sample ID",
            "gill CWA Assessment (VERIFIN) [G: green, no contamination; R: red, contamination]",
            "otoliths (TI-FI) sample ID"
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

        # for wine dataset
        # self.x_train = self.data.drop('Cultivator', axis=1)
        # self.y_train = self.data['Cultivator']

        #For Fish data
        self.x_train = self.data.drop('FDI Assessment (TI-FI) [G: green, good health status; Y: yellow, medium health status; R: red, bad health status]', axis=1)
        self.y_train = self.data['FDI Assessment (TI-FI) [G: green, good health status; Y: yellow, medium health status; R: red, bad health status]']
        
    def start(self):
        
        #training_file = 'wine_data.csv'
        training_file  = os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data.CSV'
        self.intiate_training( training_file )
        

        #testing_file  = 'wine_data_test.csv'
        #self.intiate_testing( testing_file )

        
        



