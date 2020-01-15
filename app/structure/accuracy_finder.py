
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold

from app.structure import model

class AccuracyFinder:
    def __init__(self):
        return ''

    @staticmethod
    def stratified_k_fold( mlModel, x1 ,y1 ):
        skfold = StratifiedKFold(n_splits=3, random_state=100)
        results_skfold = model_selection.cross_val_score( mlModel.get_model(), x1, y1, cv=skfold)
        return results_skfold.mean()*100.0

