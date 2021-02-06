from sklearn import model_selection
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from numpy import mean
from numpy import absolute
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow import keras
import numpy as np
from numpy import std


class AccuracyFinder:

    @staticmethod
    def stratified_k_fold(model, x, y):
        # skfold = StratifiedKFold(n_splits=5, random_state=100)

        if len(x) > 10:
            folds_split = 10
        else:
            folds_split = 2
        skfold = RepeatedKFold(n_splits=folds_split, n_repeats=3, random_state=1)
        results_skfold = model_selection.cross_val_score(model, x, y, scoring='neg_mean_absolute_error', cv=skfold, n_jobs=-1, error_score='raise')
        n_scores = absolute(results_skfold)
        print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
        return mean(n_scores)
        # return results_skfold.mean() * 100.0

    @staticmethod
    def stratified_k_fold_dnn(model, x, y, finding):
        # skfold = StratifiedKFold(n_splits=5, random_state=100)


        if len(x) > 10:
            folds_split = 10
        else:
            folds_split = 2
        kf = KFold(folds_split)
        # kf = KFold(5, shuffle=True, random_state=42)
        oos_y = []
        oos_pred = []
        fold = 0
        optimizer = keras.optimizers.Adam(lr=0.02)
        x_main, x_holdout, y_main, y_holdout = train_test_split(x, y, test_size=0.20)
        for train, test in kf.split(x):
            fold+=1
            print(f"Fold #{fold}")



            from keras import metrics
            model.compile(
                loss='mean_squared_error',
                optimizer=optimizer,
                metrics=[metrics.mean_squared_error,
                         metrics.mean_absolute_error,
                         metrics.mean_absolute_percentage_error]
                # metrics = ['accuracy']
            )
            model.fit(x_main, y_main, validation_data=(x_holdout, y_holdout), verbose=0, epochs=2)
            pred = model.predict(x_holdout)
            oos_y.append(y_holdout)
            oos_pred.append(pred)

            score = np.sqrt(metrics.mean_squared_error(pred, y_holdout))
            print(f"Fold score (RMSE): {score}")



        oos_y = np.concatenate(oos_y)
        oos_pred = np.concatenate(oos_pred)
        # oos_y_compare = np.argmax(oos_y, axis=1)
        score =  np.sqrt(metrics.mean_squared_error(oos_pred, oos_y))
        print()
        print(f"Cross-Validated score (RMSE): {score}")

        holdout_pred =model.predict(x_holdout)

        score = np.sqrt(metrics.mean_squared_error(holdout_pred, y_holdout))
        print(f"Holdout  score (accuracy): {score}")
        print("Mean: "+str(mean(score)))


        return mean(score)