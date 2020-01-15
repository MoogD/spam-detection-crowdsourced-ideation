import numpy as np
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
from Helper import importDataHelper

def create_confusionmatrix(y_true, y_pred):
    cm = ConfusionMatrix(y_true, y_pred)
    return cm


def print_confusionmatrix(cm, showstats=False):
    print("Confusion matrix:\n%s" % cm)
    if showstats:
        cm.print_stats()
    return 0


def save_confusionmatrix(cm, path, applied_filters=[], description="", dataset=""):
    cmdict = list(importDataHelper.readcsvdata(path))
    cmdict.append(cm.stats())
    cmdict[len(cmdict)-1]["applied Filter"] = applied_filters
    cmdict[len(cmdict) - 1]["Description"] = description
    cmdict[len(cmdict) - 1]["Dataset"] = dataset
    importDataHelper.writecsvfile(path, cmdict[0].keys(), cmdict)
    return 0


def load_confusionmatrices(path):
    return list(importDataHelper.readcsvdata(path))
