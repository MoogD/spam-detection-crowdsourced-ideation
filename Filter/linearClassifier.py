from os import listdir
from os.path import isfile, join

import pandas as pd
from pandas_ml import ConfusionMatrix
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt

import spamFilter
import variables
from Helper import importDataHelper

def train_linear_classifier(featurelist):
    testdata = pd.DataFrame(featurelist)
    X = testdata.drop('Spam', axis=1)
    y = testdata['Spam']
    clf = RidgeClassifier().fit(X, y)
    print(clf.score(X, y))
    return clf, clf.coef_

def train_linear_classificator(challenge, new=False):
    if new:
        unigram_tagger, st = spamFilter.prepare_tagger()
        idealist = list(
        importDataHelper.readcsvdata(variables.ideadbpath + challenge + '.csv'))
        featurelist = {}
        for idea in idealist:
            idea['TRIGGERED'] = []
            idea['PREDICTION'] = "Ham"
            idea, ideafeatures = spamFilter.classify_and_get_idea(idea, unigram_tagger, st)
            if "unusable" in idea["STATUS"] or 'spam' in idea.get("SPAM", ""):
                ideafeatures["Spam"] = 1
            else:
                ideafeatures["Spam"] = 0
            for key in ideafeatures.keys():
                featurelist[key] = featurelist.get(key, [])
                featurelist[key].append(ideafeatures[key])
    else:
        if challenge == "all":
            idealist = []
            for file in listdir(variables.linclasstrainingsdatapath):
                if isfile(join(variables.linclasstrainingsdatapath, file)):
                    idealist += list(importDataHelper.readcsvdata(join(variables.linclasstrainingsdatapath, file)))
        else:
            idealist = list(importDataHelper.readcsvdata(variables.linclasstrainingsdatapath + challenge + ".csv"))
        featurelist = {}
        for key in idealist[0].keys():
            featurelist[key] = [int(x) for x in idealist[0][key].replace('[', '').replace(']', '').split(',')]
    testdata = pd.DataFrame(featurelist)
    X = testdata.drop('Spam', axis=1)
    y = testdata['Spam']
    importDataHelper.writecsvfiledict(variables.linclasstrainingsdatapath + challenge + ".csv", featurelist.keys(), featurelist)
    clf = RidgeClassifier().fit(X, y)
    print(clf.score(X, y))
    return clf, clf.coef_

def classify(ideadict, clf):
    return clf.predict(pd.DataFrame(ideadict))[0], clf._predict_proba_lr(ideadict)[0][clf.predict(pd.DataFrame(ideadict))[0]]

def train_and_test(challenge):
    idealist = []
    if challenge == "all":
        for file in listdir(variables.linclasstrainingsdatapath):
            if isfile(join(variables.linclasstrainingsdatapath, file)):
                filename = file.split(".")[0]
                idealist += list(importDataHelper.readcsvdata(join(variables.linclasstrainingsdatapath, file)))
    else:
        idealist = list(importDataHelper.readcsvdata(variables.linclasstrainingsdatapath + challenge + ".csv"))
    featurelist = {}
    for row in idealist:
        for key in row.keys():
            featurelist[key] = featurelist.get(key, [])
            featurelist[key] += [int(x) for x in row[key].replace('[', '').replace(']', '').split(',')]
    testdata = pd.DataFrame(featurelist)
    X = testdata.drop('Spam', axis=1)
    y = testdata['Spam']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = RidgeClassifier()
    y_score = clf.fit(X_train, y_train).decision_function(X_test)
    testres = clf.predict(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in [0, 1]:
        fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color="darkorange", lw=lw, label="ROC" % roc_auc[1])
    plt.plot([0, 1], [0, 1], color="cornflowerblue", lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(challenge)
    plt.legend(loc="lower right")
    plt.savefig(variables.plotspath + "ROC_linClass_" + challenge + ".png")
    plt.show()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.show()
    confusion_matrix = ConfusionMatrix(y_test, testres)
    confusion_matrix.plot(normalized=True)
    plt.title(challenge)
    plt.savefig(variables.plotspath + "CM_linClass_" + challenge + ".png")
    plt.show()
    print(clf.coef_)
    print(classification_report(y_test, testres))
    print(confusion_matrix.stats())

