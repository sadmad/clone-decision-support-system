from sklearn import model_selection
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from numpy import mean
from app.structure import model
from numpy import absolute
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow import keras
import numpy as np


class AccuracyFinder:

    @staticmethod
    def stratified_k_fold(model, x, y):
        # skfold = StratifiedKFold(n_splits=5, random_state=100)
        skfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        results_skfold = model_selection.cross_val_score(model, x, y, scoring='neg_mean_absolute_error', cv=skfold, n_jobs=-1, error_score='raise')
        n_scores = absolute(results_skfold)
        return mean(n_scores)
        # return results_skfold.mean() * 100.0

    @staticmethod
    def stratified_k_fold_dnn(model, x, y, finding):
        # skfold = StratifiedKFold(n_splits=5, random_state=100)

        kf = KFold(10)
        # kf = KFold(5, shuffle=True, random_state=42)
        oos_y = []
        oos_pred = []
        fold = 0
        columns_x = len(finding.x_train[0])
        # columns_x = len(finding.x_train[0])
        columns_y = len(finding.y_train.columns)
        output_neuron_c = 3
        # output_neuron_r = columns_yasaaaaaaaaaaaaaa
        activation_function = 'relu'
        output_activation_function_r = 'relu'
        output_activation_function__c = 'relu'
        optimizer = keras.optimizers.Adam(lr=0.02)
        x_main, x_holdout, y_main, y_holdout = train_test_split(x, y, test_size=0.20)
        for train, test in kf.split(x):
            fold+=1
            print(f"Fold #{fold}")


            Output_Layer = len(set(finding.y_train))
            Input_layer = keras.Input(shape=(columns_x,))
            Dense_Layers = keras.layers.Dense(500, activation='relu')(Input_layer)
            Dense_Layers = keras.layers.Dense(256, activation='relu')(Dense_Layers)
            Dense_Layers = keras.layers.Dense(128, activation='relu')(Dense_Layers)
            Dense_Layers = keras.layers.Dense(500, activation='relu')(Dense_Layers)
            Dense_Layers = keras.layers.Dense(1000, activation='relu')(Dense_Layers)

            # Dense_Layers = Concatenate()([Input_layer, Dense_Layers])
            Dense_Layers = keras.layers.Dense(Output_Layer, activation='linear')(Dense_Layers)
            modelReg = keras.Model(inputs=Input_layer, outputs=Dense_Layers)

            # out1 = Dense(1)(Dense_Layers)
            # out2 = Dense(1)(Dense_Layers)

            # modelReg = Model(inputs = Input_layer, outputs = [out1,out2])
            from keras import metrics
            modelReg.compile(
                loss='mean_squared_error',
                optimizer=optimizer,
                metrics=[metrics.mean_squared_error,
                         metrics.mean_absolute_error,
                         metrics.mean_absolute_percentage_error]
                # metrics = ['accuracy']
            )
            modelReg.fit(x_main, y_main, validation_data=(x_holdout, y_holdout), verbose=0, epochs=2)
            pred = modelReg.predict(x_holdout)
            oos_y.append(y_holdout)
            oos_pred.append(pred)

            score = np.sqrt(metrics.mean_squared_error(pred, y_holdout))
            print(f"Fold score (RMSE): {score}")

            # y_compare = np.argmax(pred, axis=1)
            # score = metrics.accuracy_score(y_compare, pred)
            # print(f"Fold score (accuracy): {score}")

        oos_y = np.concatenate(oos_y)
        oos_pred = np.concatenate(oos_pred)
        # oos_y_compare = np.argmax(oos_y, axis=1)
        score =  np.sqrt(metrics.mean_squared_error(oos_pred, oos_y))
        print()
        print(f"Cross-Validated score (RMSE): {score}")

        holdout_pred =modelReg.predict(x_holdout)

        score = np.sqrt(metrics.mean_squared_error(holdout_pred, y_holdout))
        print(f"Holdout  score (accuracy): {score}")
        print("Mean: "+str(mean(score)))
        # score = metrics.accuracy_score(y_compare, pred)
        # print(f"Final,  score (accuracy): {score}")

        # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        # model.fit(X_train, y_train, batch_size = 10, epochs = 100)
        # y_pred = model.predict(X_test)
        # y_pred = (y_pred > 0.5)
        # cm = confusion_matrix(y_test, y_pred)
        # classiifier = KerasClassifier(build_fn=model,
        #                               batch_size=10, nb_epoch=100)
        # accuracies = cross_val_score(estimator=model,
        #                              X=X_train,
        #                              y=y_train,
        #                              cv=10,
        #                              n_jobs=-1)
        # scores = model.evaluate(X_train, y_train, verbose=0)
        # n_scores = absolute(scores)
        # mean = 0 #accuracies.mean()
        # variance = accuracies.var()



        # return mean(n_scores)
        # return results_skfold.mean() * 100.0
