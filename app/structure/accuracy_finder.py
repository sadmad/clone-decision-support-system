from sklearn import model_selection
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from numpy import mean
from app.structure import model
from numpy import absolute


class AccuracyFinder:

    @staticmethod
    def stratified_k_fold(model, x, y):
        # skfold = StratifiedKFold(n_splits=5, random_state=100)
        skfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        results_skfold = model_selection.cross_val_score(model, x, y, scoring='neg_mean_absolute_error', cv=skfold, n_jobs=-1, error_score='raise')
        n_scores = absolute(results_skfold)
        return mean(n_scores)
        # return results_skfold.mean() * 100.0
