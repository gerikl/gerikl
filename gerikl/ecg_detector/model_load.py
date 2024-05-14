import pickle
import typing as tp

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def load_model(model: str = "logreg") -> tp.Union[LogisticRegression, RandomForestClassifier]:
    """
    load pretrained model. Available options are logreg and randforest. Default is logreg.
    """
    if model == "logreg":
        with open('logreg_main.pkl', 'rb') as f:
            clf = pickle.load(f)
    elif model == "randforest":
        with open("RandForest_good.pkl", "rb") as f:
            clf = pickle.load(f)
    return clf
