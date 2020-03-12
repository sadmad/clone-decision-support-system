
from app.structure import dss
from app import app
import os
import pandas as pd

class ML_Model:

    def __init__(self):
        self.__name = None
        self.__model = None
        self.__accuracy = None
        self.saved_model_scaler = os.path.join(app_path, obj['scaler'])
        self.saved_model_path = os.path.join(app_path, obj['model'])

    def set_name(self, name):
        self.__name = name
    def set_model(self, model):
        self.__model = model
    def set_accuracy(self, accuracy):
        self.__accuracy = accuracy


    def get_model(self):
        return self.__model
    def get_name(self):
        return self.__name
    def get_accuracy(self):
        return self.__accuracy



class Fish:

    def __init__(self, model_type, model_name):
        
        #Data Members
        self._features = [
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


        self._features = [
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
        self._DSS                     = None
        self._model_config            = None
        self._train_data              = None
        self._x_train                 = None
        self._y_train                 = None


        model_storage_folder = os.path.join(app.root_path, 'storage/models')

        if (model_type == 1):

            self._DSS           = dss.NeuralNetwork()
            self._model_config  = app.config['NEURAL_NETWORK_MODEL']

        elif (model_type == 2):

            self._DSS           = dss.RandomForest()
            self._model_config  = app.config['RANDOM_FOREST_CLASSIFIER_MODEL']

        elif (model_type == 3):

            self._DSS           = dss.LinearRegressionM()
            self._model_config  = app.config['LINEAR_REGRESSION_MODEL']

        elif (model_type == 4):
            
            self._DSS           = dss.LogisticRegressionM()
            self._model_config  = app.config['LOGISTIC_REGRESSION_MODEL']


        if not os.path.exists(model_storage_folder):
            os.makedirs(model_storage_folder)

        self._DSS.trained_model_scaler  = os.path.join( model_storage_folder, model_name + self._model_config['scaler'] )
        self._DSS.trained_model         = os.path.join( model_storage_folder, model_name + self._model_config['model'] )

    def triger_dss_training( self ):
        #plan 0> pass object of fish into dss . . do not use x,y or any other attribute in dsss. . jsut pass from here to dss
        # pass self(x,y,paths etc) to dss, 
        self._DSS.data_preprocessing( )
        self._DSS.training()
        print('awais')
        #modelObject = DSS.training()

class FDI_ASSESMENT(Fish):
    
    def __init__(self, model_type):
        
        super().__init__( model_type, app.config['MODELS']['FDI_ASSESMENT'] )

    def start(self):

        print(' FDI_ASSESMENT Data_intialization')
        
        #self._DSS.train_data = pd.read_csv( os.path.dirname(os.path.dirname(__file__)) + '/data/fish/DAIMON_Cod_Data.CSV' , names = self._features)
        self._DSS.train_data = pd.read_csv('wine_data.csv', names=self._features)
        self._DSS.x_train = self._DSS.train_data.drop('Cultivator', axis=1)
        self._DSS.y_train = self._DSS.train_data['Cultivator']
        return super().triger_dss_training()
        return 'awais'



