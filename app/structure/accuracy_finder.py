
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold

from app.structure import model

class AccuracyFinder:

    @staticmethod
    def stratified_k_fold( model, x ,y ):
        skfold = StratifiedKFold(n_splits=5, random_state=100)
        results_skfold = model_selection.cross_val_score( model, x, y, cv=skfold)
        return results_skfold.mean()*100.0

