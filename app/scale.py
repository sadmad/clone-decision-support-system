from sklearn.preprocessing import StandardScaler
import os
import joblib




class Scale:
    @staticmethod
    def StandardScaler( X_train, ScalerObjPath ):
       
        scaler = StandardScaler()      
        scaler.fit(X_train)
        #https://stackoverflow.com/questions/41993565/save-minmaxscaler-model-in-sklearn

        if os.path.exists(ScalerObjPath):
            os.remove(ScalerObjPath)
        joblib.dump(scaler, ScalerObjPath ) 

        return scaler.transform(X_train)



    @staticmethod
    def LoadScalerTest( X_test, ScalerObjPath ):
       
        scaler = joblib.load( ScalerObjPath )
        return scaler.transform(X_test)
    

    @staticmethod
    def LoadScalerAndScaleTestData( X_test, ScalerObjPath ):
       
        scaler = joblib.load( ScalerObjPath )
        return scaler.transform(X_test)


