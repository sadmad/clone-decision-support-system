from app import app
from sklearn.preprocessing import StandardScaler
import os
import joblib


app_path = app.root_path


class Scale:
    @staticmethod
    def StandardScaler( X_train ):
       
        scaler = StandardScaler()      
        scaler.fit(X_train)
        #https://stackoverflow.com/questions/41993565/save-minmaxscaler-model-in-sklearn
        file_path = os.path.join(app_path,app.config['SCALER_FILENAME'] )
        joblib.dump(scaler, file_path ) 

        return scaler.transform(X_train)



    @staticmethod
    def LoadScalerTest( X_test ):
       
        file_path = os.path.join(app_path,app.config['SCALER_FILENAME'] )
        scaler = joblib.load( file_path )
        return scaler.transform(X_test)


