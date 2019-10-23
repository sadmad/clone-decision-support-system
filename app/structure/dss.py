import pandas as pd
from app import app
from app import scale
import os
from sklearn.neural_network import MLPClassifier
import joblib
from flask import  jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
#import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


app_path = os.path.join(app.root_path,'models' )

features = [
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

class DSS:
    def __init__(self, obj = None):

        self.train_data = None
        self.x_train    = None
        self.y_train    = None
        self.model_key    = obj['model']
        self.saved_model_scaler  = os.path.join(app_path, obj['scaler'] )
        self.saved_model_path    = os.path.join(app_path, self.model_key )
        
        self.test_data  = None
        self.x_test     = None
        self.y_test     = None

    def data_intialization(self):

        self.train_data = pd.read_csv('wine_data_train.csv', names = features)
        self.x_train    = self.train_data.drop('Cultivator',axis=1)
        self.y_train    = self.train_data['Cultivator']

    def data_preprocessing(self):
        self.x_train = scale.Scale.StandardScaler( self.x_train,self.saved_model_scaler )

    def save_model(self,model = None):

            
            if os.path.exists(self.saved_model_path):
                os.remove(self.saved_model_path)

            # save the model to disk
            #https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

            #pickle.dump(model, open(app.config['NEURAL_NETWORK_MODEL'], 'wb'))
            joblib.dump(model, self.saved_model_path)


    def testing(self):
        
        self.test_data  = pd.read_csv('wine_data_test.csv',  names = features)
        self.x_test  = self.test_data.drop('Cultivator',axis=1)
        self.y_test  = self.test_data['Cultivator']

        self.x_test = scale.Scale.LoadScalerAndScaleTestData( self.x_test, self.saved_model_scaler )

        # load the model from disk
        #loaded_model = pickle.load(open(saved_model_path, 'rb'))
        loaded_model = joblib.load(self.saved_model_path)
        score_result = loaded_model.score(self.x_test, self.y_test)
        predictions  = loaded_model.predict(self.x_test)
        # print(confusion_matrix(self.y_test,predictions))
        # print(classification_report(self.y_test,predictions))

        return jsonify([{
            'status':200,
            'message':'Test Obervations are predicted by Neural Network Trained Model.',
            'predictions' : pd.Series(predictions).to_json(orient='values')
        }])

class NeuralNetwork(DSS):

    def training(self):

        model = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
        model.fit(self.x_train,self.y_train)
        return model

class RandomForest(DSS):

    def training(self):

        #X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
        model.fit(self.x_train,self.y_train)  
        return model


class LinearRegressionM(DSS):

    def training(self):

        model = LinearRegression()      
        model.fit(self.x_train, self.y_train) #training the algorithm  
        return model


class LogisticRegressionM(DSS):

    def training(self):

        print('LogisticRegression')
        model = LogisticRegression()      
        model.fit(self.x_train, self.y_train) #training the algorithm  
        return model
        