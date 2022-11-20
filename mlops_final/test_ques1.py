from ques1 import preprocess_digits, run_multiple_models
from sklearn import datasets, metrics
from sklearn.svm import SVC
import numpy as np


def test_quest1():

    parameters = {
    'svc': {'model': SVC, 'params': {'kernel': ['rbf'], 'gamma': [0.01], 'C': [0.1]}},
    }
    
    no_of_splits = 5
    train_frac, dev_frac, test_frac = 0.8, 0.1 , 0.1
    metric=metrics.accuracy_score

    digits = datasets.load_digits()
    data, label = preprocess_digits(digits)
    results = run_multiple_models(data, label, parameters, no_of_splits, train_frac, dev_frac, test_frac, metric)
    assert results[0]==results[1]
