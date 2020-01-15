import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from Filter import bayes, complexBayes, duplicateDetection, linearClassifier, USEClassifier
from Helper import importDataHelper, evaluationHelper
from Visualization import confusionMatrix
from sklearn.model_selection import train_test_split
import variables
import spamFilter


def evaluate_system_use(X_train, X_test, challenge=None, dups=False, plotname=None):
    data = {"DESCRIPTION": [], "Spam": []}
    for idea in X_train:
        data["DESCRIPTION"].append(idea["DESCRIPTION"])
        if idea["STATUS"] == "unusable":
            data["Spam"].append(1)
        elif idea["STATUS"] == "usable":
            data["Spam"].append(0)
        elif "spam" in idea["SPAM"]:
            data["Spam"].append(1)
        else:
            data["Spam"].append(0)

    print("prepared")
    if dups:
        useest = USEClassifier.train_classifier_idealist_use(pd.DataFrame(data),
                                                         "Data/firstTestrun/KlassifikatorData/" + plotname + "/Dups/")
    else:
        useest = USEClassifier.train_classifier_idealist_use(pd.DataFrame(data), "Data/firstTestrun/KlassifikatorData/" + plotname + "/")



    actual = []
    predUSE = []
    for idea in X_test:
        if not idea["STATUS"] == "unreviewed":
            actual.append(idea["STATUS"] == "unusable")
        else:
            actual.append("spam" in idea["SPAM"])
        if idea["DESCRIPTION"] == "" or idea["DESCRIPTION"] == []:
            predUSE.append((0, 0.0))
        else:
            predUSE.append((USEClassifier.classify(useest, {"DESCRIPTION": idea["DESCRIPTION"]})))
        idea["TRIGGERED"] = idea.get("TRIGGERED", [])


    results = {"actual": actual, "USE": predUSE}
    importDataHelper.writecsvfiledict("Data/firstTestrun/Results/USEtest/evaluationResults.csv", results.keys(), results)
    return results


def eval_USE():
    challengedict = {
        "TCO": list(importDataHelper.readcsvdata(variables.firstrunpath + variables.ideadbpath + "TCO.csv")),
        "bionicRadar": list(
            importDataHelper.readcsvdata(variables.firstrunpath + variables.ideadbpath + "bionicRadar.csv")),
        "fabricDisplay": list(
            importDataHelper.readcsvdata(variables.firstrunpath + variables.ideadbpath + "fabricDisplay.csv"))}
    # Create Dict for numbers of filtered Duplicates in each Test
    dupdict = {}
    # Start Test for each Challenge
    for key in challengedict.keys():
        # Create List with ideas from other challenges to extend Trainingsdataset:
        idealist = []
        for key2 in challengedict.keys():
            if key2 is not key:
                idealist += challengedict[key2].copy()
        X_train = list(importDataHelper.readcsvdata(variables.firstrunpath + variables.trainingideaspath + key + ".csv"))
        X_test = list(importDataHelper.readcsvdata(variables.firstrunpath + variables.testideapath + "BeforeClassification" + key + ".csv"))
        X_ndtrain = duplicateDetection.filterduplikates(X_train,
                                                    "Data/firstTestrun/DuplicateResults/Trainingsdata/" + key + ".csv")
        dupdict[key] = len(X_train) - len(X_ndtrain)

        # Create Copys of Testset to avoid Changes in Testset while Testing
        X_ndtest = X_test.copy()
        idealist_test = X_test.copy()
        idealist_nodups_test = X_test.copy()

        # Train and test Classifier without duplicate detection and Extension
        results = evaluate_system_use(X_train, X_test, challenge=key, plotname=key)
        importDataHelper.writecsvfiledict(variables.firstrunpath + variables.testresultpath + key + ".csv",
                                          results.keys(), results)
        print("Done first set")

        # Train and test Classifier with duplicate detection and without Extension
        results2 = evaluate_system_use(X_ndtrain, X_ndtest, key, dups=True, plotname=key)
        importDataHelper.writecsvfiledict("Data/firstTestrun/Results/USEtest/evaluationResultsNoDups" + key + ".csv",
                                          results2.keys(), results2)
        print("Challenge training done", key)

        # remove Duplicate from extended Trainingsset
        idealist_nodups = duplicateDetection.filterduplikates(idealist, variables.resultpath + "eval" + key + ".csv")
        dupdict[key + " All"] = len(idealist) - len(idealist_nodups)

        # Train and test Classifier without duplicate detection and with Extension
        results = evaluate_system_use(idealist, idealist_test, plotname=key + "Erweitert")
        importDataHelper.writecsvfiledict("Data/firstTestrun/Results/USEtest/evaluationAll" + key + ".csv",
                                          results.keys(), results)
        print("Done first set")
        # Train and test Classifier with duplicate detection and Extension
        results2 = evaluate_system_use(idealist_nodups, idealist_nodups_test, dups=True, plotname=key + "Erweitert")
        importDataHelper.writecsvfiledict("Data/firstTestrun/Results/USEtest/evaluationResultsNoDupsAll" + key + ".csv",
                                          results2.keys(), results2)
        print("All training done", key)
    print(dupdict)
    importDataHelper.writecsvfiledict("Data/firstTestrun/Results/USEtest/dupNums.csv", dupdict.keys(), dupdict)


def eval_newAll():
    # Import all Ideas divided by Challenge:
    challengedict = {"TCO": list(importDataHelper.readcsvdata(variables.firstrunpath + variables.ideadbpath +"TCO.csv")),
                     "bionicRadar": list(importDataHelper.readcsvdata(variables.firstrunpath + variables.ideadbpath + "bionicRadar.csv")),
                     "fabricDisplay": list(importDataHelper.readcsvdata(variables.firstrunpath + variables.ideadbpath + "fabricDisplay.csv"))}
    # Create Dict for numbers of filtered Duplicates in each Test
    dupdict = {}
    # Start Test for each Challenge
    for key in challengedict.keys():
        # Create List with ideas from other challenges to extend Trainingsdataset:
        idealist = []
        for key2 in challengedict.keys():
            if key2 is not key:
                idealist += challengedict[key2].copy()
        # Split Ideas for this challenge in test and traindataset
        X_train, X_test = train_test_split(challengedict[key], test_size=0.33)
        # Save Trainingsdata (with duplicate and without extension)
        importDataHelper.writecsvfile(variables.firstrunpath + variables.trainingideaspath + key + ".csv",
                                          X_train[0].keys(), X_train)
        # Save Testdata
        importDataHelper.writecsvfile(variables.firstrunpath + variables.testideapath +
                                          "BeforeClassification" + key + ".csv", X_test[0].keys(), X_test)
        print("Saved Data for", key)

        idealist += X_train.copy()

        # Create Trainings list without duplicate
        X_ndtrain = duplicateDetection.filterduplikates(X_train, "Data/firstTestrun/DuplicateResults/Trainingsdata/" + key + ".csv")
        dupdict[key] = len(X_train) - len(X_ndtrain)

        # Create Copys of Testset to avoid Changes in Testset while Testing
        X_ndtest = X_test.copy()
        idealist_test = X_test.copy()
        idealist_nodups_test = X_test.copy()

        # Train and test Classifier without duplicate detection and Extension
        results = evaluate_system(X_train, X_test, challenge=key, plotname=key)
        importDataHelper.writecsvfiledict(variables.firstrunpath + variables.testresultpath + key + ".csv", results.keys(), results)
        print("Done first set")

        # Train and test Classifier with duplicate detection and without Extension
        results2 = evaluate_system(X_ndtrain, X_ndtest, key, dups=True, plotname=key)
        importDataHelper.writecsvfiledict("Data/ResultsAllNew/evaluationResultsNoDups" + key + ".csv", results2.keys(),
                                          results2)
        print("Challenge training done", key)

        # remove Duplicate from extended Trainingsset
        idealist_nodups = duplicateDetection.filterduplikates(idealist, variables.resultpath + "eval" + key + ".csv")
        dupdict[key + " All"] = len(idealist) - len(idealist_nodups)

        # Train and test Classifier without duplicate detection and with Extension
        results = evaluate_system(idealist, idealist_test, plotname=key + "Erweitert")
        importDataHelper.writecsvfiledict("Data/ResultsAllNew/evaluationAll" + key + ".csv", results.keys(), results)
        print("Done first set")
        # Train and test Classifier with duplicate detection and Extension
        results2 = evaluate_system(idealist_nodups, idealist_nodups_test, dups=True, plotname=key + "Erweitert")
        importDataHelper.writecsvfiledict("Data/ResultsAllNew/evaluationResultsNoDupsAll" + key + ".csv",
                                          results2.keys(),
                                          results2)
        print("All training done", key)
    print(dupdict)
    importDataHelper.writecsvfiledict("Data/ResultsAllNew/dupNums.csv", dupdict.keys(), dupdict)

def eval_all():
    challengedict = {"TCO": list(importDataHelper.readcsvdata("Data/DBs/ideaDB/TCO.csv")), "bionicRadar": list(importDataHelper.readcsvdata("Data/DBs/ideaDB/bionicRadar.csv")), "fabricDisplay": list(importDataHelper.readcsvdata("Data/DBs/ideaDB/fabricDisplay.csv"))}
    dupdict = {}
    for key in challengedict.keys():
        idealist = []
        for key2 in challengedict.keys():
            if key2 is not key:
                idealist += challengedict[key2].copy()
        X_train, X_test = train_test_split(challengedict[key], test_size=0.33)
        idealist += X_train.copy()

        X_ndtrain = duplicateDetection.filterduplikates(X_train, variables.resultpath + "eval2" + key + ".csv")
        dupdict[key] = len(X_train) - len(X_ndtrain)
        X_ndtest = X_test.copy()
        idealist_test = X_test.copy()
        idealist_nodups_test = X_test.copy()
        results = evaluate_system(X_train, X_test, key)
        importDataHelper.writecsvfiledict("Data/ResultsAllNew/evaluation" + key + ".csv", results.keys(), results)
        print("Done first set")
        results2 = evaluate_system(X_ndtrain, X_ndtest, key, dups=True)
        importDataHelper.writecsvfiledict("Data/ResultsAllNew/evaluationResultsNoDups" + key + ".csv", results2.keys(),
                                          results2)
        print("Challenge training done", key)

        idealist_nodups = duplicateDetection.filterduplikates(idealist, variables.resultpath + "eval" + key + ".csv")
        dupdict[key + " All"] = len(idealist) - len(idealist_nodups)
        results = evaluate_system(idealist, idealist_test)
        importDataHelper.writecsvfiledict("Data/ResultsAllNew/evaluationAll" + key + ".csv", results.keys(), results)
        print("Done first set")
        results2 = evaluate_system(idealist_nodups, idealist_nodups_test, dups=True)
        importDataHelper.writecsvfiledict("Data/ResultsAllNew/evaluationResultsNoDupsAll" + key + ".csv", results2.keys(),
                                          results2)
        print("All training done", key)
    print(dupdict)
    importDataHelper.writecsvfiledict("Data/ResultsAllNew/dupNums.csv", dupdict.keys(), dupdict)

def evaluate_fun():
    idealist = list(importDataHelper.readcsvdata("Data/DBs/ideaDB/bionicRadar.csv"))
    X_train, X_test = train_test_split(idealist, test_size=0.33)
    X_ndtrain = duplicateDetection.filterduplikates(X_train, variables.resultpath + "evalbionicRadar.csv")
    X_ndtest = X_test.copy()
    results = evaluate_system(X_train, X_test, "bionicRadar")
    importDataHelper.writecsvfiledict("Data/ResultsNew/evaluationResultsbionicRadar.csv", results.keys(), results)
    print("Done first set")
    results2 = evaluate_system(X_ndtrain, X_ndtest, "bionicRadar",dups=True)
    importDataHelper.writecsvfiledict("Data/ResultsNew/evaluationResultsNoDupsbionicRadar.csv", results2.keys(), results2)

    print("Done")


def evaluate_system(X_train, X_test, challenge=None, dups=False, plotname=None):
    # prepare filtersystem
    unigram_tagger, st = spamFilter.prepare_tagger()

    # prepare data for the linear classifier (features) and neural net (data)
    features = {}
    data = {"DESCRIPTION": [], "Spam": []}
    for idea in X_train:
        idea, feature = spamFilter.classify_and_get_idea(idea, unigram_tagger, st)
        data["DESCRIPTION"].append(idea["DESCRIPTION"])
        for key in feature.keys():
            features[key] = features.get(key, [])
            features[key].append(feature[key])
        if idea["STATUS"] == "unusable":
            features["Spam"] = features.get("Spam", [])
            features["Spam"].append(1)
            data["Spam"].append(1)
        elif idea["STATUS"] == "usable":
            features["Spam"] = features.get("Spam", [])
            features["Spam"].append(0)
            data["Spam"].append(0)
        elif "spam" in idea["SPAM"]:
            features["Spam"] = features.get("Spam", [])
            features["Spam"].append(1)
            data["Spam"].append(1)
        else:
            features["Spam"] = features.get("Spam", [])
            features["Spam"].append(0)
            data["Spam"].append(0)

    print("prepared")
    # Train Bayes classificators
    bayes.trainbayes(X_train, challenge=challenge, delete=True, duplicates=dups)
    complexBayes.trainbayes(X_train, challenge=challenge, delete=True,  duplicates=dups)

    wordprobs = bayes.gettokenprobs(challenge=challenge,  duplicates=dups)
    comwordprobs = complexBayes.gettokenprobs(challenge=challenge,  duplicates=dups)
    print("Bayes Training done")

    linClass, coeff = linearClassifier.train_linear_classifier(features)
    # Plot ROC for Trainingsdata
    if plotname is not None:
        plot_lin_Classifier(features["Spam"], linClass.decision_function(pd.DataFrame(features).drop('Spam', axis=1)), plotname, dups)

    if dups:
        useest = USEClassifier.train_classifier_idealist(pd.DataFrame(data),
                                                         "Data/firstTestrun/KlassifikatorData/" + plotname + "/Dups/")
    else:
        useest = USEClassifier.train_classifier_idealist(pd.DataFrame(data), "Data/firstTestrun/KlassifikatorData/" + plotname + "/")

    print(coeff)

    actual = []
    predbay = []
    predcombay = []
    predUSE = []
    predLin = []
    features = {}
    for idea in X_test:
        if not idea["STATUS"] == "unreviewed":
            actual.append(idea["STATUS"] == "unusable")
        else:
            actual.append("spam" in idea["SPAM"])
        predbay.append(bayes.classify(idea["DESCRIPTION"], wordprobs))
        predcombay.append(complexBayes.classify(idea["DESCRIPTION"], comwordprobs))
        if idea["DESCRIPTION"] == "" or idea["DESCRIPTION"] == []:
            predUSE.append((0, 0.0))
        else:
            predUSE.append((USEClassifier.classify(useest, {"DESCRIPTION": idea["DESCRIPTION"]})))
        idea["TRIGGERED"] = idea.get("TRIGGERED", [])
        idea, ideadata = spamFilter.classify_and_get_idea(idea, unigram_tagger, st)
        test = False
        for ideakey in ideadata.keys():
            if ideadata[ideakey] == 1:
                features[ideakey] = features.get(ideakey, [])
                features[ideakey].append(1)
                test = True
            else:
                features[ideakey] = features.get(ideakey, [])
                features[ideakey].append(0)
            ideadata[ideakey] = [ideadata[ideakey]]

        if test:
            predLin.append(linearClassifier.classify(pd.DataFrame(ideadata), linClass))
        else:
            predLin.append((0, 0.0))
    results = {"actual": actual, "bayes": predbay, "complexbayes": predcombay, "USE": predUSE, "linCLassifier": predLin,
               "linClassCo": coeff, "Filter": features}
    importDataHelper.writecsvfiledict(variables.resultpath + "evaluationResults.csv", results.keys(), results)
    return results


def evaluate_results():
    resultdict = import_results()

    safelist = []
    for key in resultdict.keys():
        print(key)
        print("Ideas: ", len(resultdict[key]["actual"]))
        print("Spam: ", resultdict[key]["actual"].count(True))
        print("Ham: ", resultdict[key]["actual"].count(False))
        bayespred = [x >= 0.9 for x in resultdict[key]["bayes"]]
        bayesprob = [x for x in resultdict[key]["bayes"]]
        combayespred = [x >= 0.9 for x in resultdict[key]["complexbayes"]]
        combayesprob = [x for x in resultdict[key]["complexbayes"]]
        linclasspred = [x[0] == 1 for x in resultdict[key]["linCLassifier"]]
        linclassprob = [x[1] for x in resultdict[key]["linCLassifier"]]
        filterpred = [x == 1 for x in evaluationHelper.get_filter_results(resultdict[key]["Filter"])]
        usepred = [x[0] == 1 for x in resultdict[key]["USE"]]
        useprob = [x[1] for x in resultdict[key]["USE"]]
        lin = False
        com = False
        use = False
        bay = False
        if True in bayespred and False in bayespred:
            cmbay = confusionMatrix.create_confusionmatrix(resultdict[key]["actual"], bayespred)
            safelist.append({"Data": key, "Filter": "Bayes", **cmbay.stats()})
            bay = True
            print("Bayes")
            print("Precision: ", cmbay.PPV)
            print("Recall: ", cmbay.TPR, "\n")
        if True in combayespred and False in combayespred:
            cmcombay = confusionMatrix.create_confusionmatrix(resultdict[key]["actual"], combayespred)
            safelist.append({"Data": key, "Filter": "complex Bayes", **cmcombay.stats()})
            com = True
            print("Complex Bayes")
            print("Precision: ", cmcombay.PPV)
            print("Recall: ", cmcombay.TPR, "\n")
        if True in linclasspred and False in linclasspred:
            cmlinclass = confusionMatrix.create_confusionmatrix(resultdict[key]["actual"], linclasspred)
            safelist.append({"Data": key, "Filter": "lin Classifier", **cmlinclass.stats()})
            lin = True
            print("lin Classifier")
            print("Precision: ", cmlinclass.PPV)
            print("Recall: ", cmlinclass.TPR, "\n")
        if True in filterpred and False in filterpred:
            cmfilter = confusionMatrix.create_confusionmatrix(resultdict[key]["actual"], filterpred)
            safelist.append({"Data": key, "Filter": "Filtersystem", **cmfilter.stats()})
            print("Filtersystem")
            print("Precision: ", cmfilter.PPV)
            print("Recall: ", cmfilter.TPR, "\n")
        if True in usepred and False in usepred:
            cmuse = confusionMatrix.create_confusionmatrix(resultdict[key]["actual"], usepred)
            safelist.append({"Data": key, "Filter": "USE", **cmuse.stats()})
            use = True
            print("USE Classifier")
            print("Precision: ", cmuse.PPV)
            print("Recall: ", cmuse.TPR, "\n")
        probs = []
        classor = []
        classtwo = []
        classthree = []
        countbayesdiff = 0
        y = 0
        for i in range(0,len(bayesprob)):
            classor.append(bayesprob[i] >= 0.9 or combayesprob[i] >= 0.9 or linclasspred[i] or usepred[i])
            classtwo.append((bayesprob[i] >= 0.9 and(combayesprob[i] >= 0.9 or linclasspred[i] or usepred[i])) or
                            (combayesprob[i] >= 0.9 and (linclasspred[i] or usepred[i])) or (linclasspred[i] and usepred[i]))
            classthree.append((bayesprob[i] >= 0.9 and combayesprob[i] >= 0.9 and (linclasspred[i] or usepred[i])) or
                              (combayesprob[i] >= 0.9 and linclasspred[i] and usepred[i]) or
                              (bayesprob[i] >= 0.9 and linclasspred[i] and usepred[i]))
            probs.append(0.0)
            if bay:
                probs[i] += bayesprob[i]
                y += 1
            if com:
                probs[i] += combayesprob[i]
                y += 1
            if lin:
                if linclasspred[i]:
                    probs[i] += linclassprob[i]
                else:
                    probs[i] += 1-linclassprob[i]
                y += 1
            if use:
                if usepred[i]:
                    probs[i] += useprob[i]
                else:
                    probs[i] += 1 - useprob[i]
                y += 1
            if y > 0:
                probs[i] = probs[i]/y
            if bayesprob[i] >= 0.9 and combayesprob[i] < 0.9:
                countbayesdiff += 1
        print("Bayes difference: ", countbayesdiff, "\n\n")
        avglow = [x >= 0.5 for x in probs]
        avghigh = [x >= 0.8 for x in probs]
        if True in avglow and False in avglow:
            cmavglow = confusionMatrix.create_confusionmatrix(resultdict[key]["actual"], avglow)
            safelist.append({"Data": key, "Filter": "low avg", **cmavglow.stats()})
            print("low Average")
            print("Precision: ", cmavglow.PPV)
            print("Recall: ", cmavglow.TPR, "\n")

        if True in avghigh and False in avghigh:
            cmavghigh = confusionMatrix.create_confusionmatrix(resultdict[key]["actual"], avghigh)
            safelist.append({"Data": key, "Filter": "high avg", **cmavghigh.stats()})
            print("high Average")
            print("Precision: ", cmavghigh.PPV)
            print("Recall: ", cmavghigh.TPR, "\n")
        if True in classor and False in classor:
            cmor = confusionMatrix.create_confusionmatrix(resultdict[key]["actual"], classor)
            safelist.append({"Data": key, "Filter": "Or Classifiers", **cmor.stats()})
            print("Classifier or")
            print("Precision: ", cmor.PPV)
            print("Recall: ", cmor.TPR, "\n")
        if True in classtwo and False in classtwo:
            cmtwo = confusionMatrix.create_confusionmatrix(resultdict[key]["actual"], classtwo)
            safelist.append({"Data": key, "Filter": "Two Classifiers", **cmtwo.stats()})
            print("Two Classifier")
            print("Precision: ", cmtwo.PPV)
            print("Recall: ", cmtwo.TPR, "\n")
        if True in classthree and False in classthree:
            cmthree = confusionMatrix.create_confusionmatrix(resultdict[key]["actual"], classthree)
            safelist.append({"Data": key, "Filter": "Three Classifiers", **cmthree.stats()})
            print("Three Classifier")
            print("Precision: ", cmthree.PPV)
            print("Recall: ", cmthree.TPR, "\n")
    sortedDict = {"Bayes": [], "complex Bayes": [], "lin Classifier": [], "Filtersystem": [], "USE": [],
                  "Or Classifiers": [], "Two Classifiers": [], "Three Classifiers": [], "low avg": []}
    for result in safelist:
        sortedDict[result["Filter"]].append({"Data": result["Data"],
                                             "population": result["population"],
                                             "TN": result["TN"], "FP": result["FP"], "N": result["N"],
                                             "FN": result["FN"], "TP": result["TP"], "P": result["P"],
                                             "PositiveTest": result["PositiveTest"],
                                             "NegativeTest": result["NegativeTest"],
                                             "TPR": result["TPR"], "PPV": result["PPV"], "Acc": result["ACC"]})

    for key in sortedDict:
        if len(sortedDict[key]) > 0:
            importDataHelper.writecsvfile(variables.firstrunpath + variables.testresultpath + "first/" + key + "ResultDict.csv",
                                      sortedDict[key][0].keys(), sortedDict[key])

    importDataHelper.writecsvfile(variables.firstrunpath + variables.testresultpath + "first/ResultDicts.csv", safelist[0].keys(), safelist)

def plot_bayes_results():
    resultdict = import_results()
    for key in resultdict.keys():
        compreclist = []
        comreclist = []
        baypreclist = []
        bayreclist = []
        for i in range(100, 0, -1):
            combayespred = [x >= 1/i for x in resultdict[key]["complexbayes"]]
            bayespred = [x >= 1/i for x in resultdict[key]["bayes"]]
            if True in combayespred and False in combayespred:
                cmcom = confusionMatrix.create_confusionmatrix(resultdict[key]["actual"], combayespred)
                compreclist.append(cmcom.PPV)
                comreclist.append(cmcom.TPR)
            else:
                compreclist.append(0.0)
                comreclist.append(0.0)

            if True in bayespred and False in bayespred:
                cmbay = confusionMatrix.create_confusionmatrix(resultdict[key]["actual"], bayespred)
                baypreclist.append(cmbay.PPV)
                bayreclist.append(cmbay.TPR)
            else:
                baypreclist.append(0.0)
                bayreclist.append(0.0)

        if "fabric" in key:
            dataset = "Fabric Display"
        elif "bionic" in key:
            dataset = "Bionic Radar"
        else:
            dataset = "TCO"

        if "All" in key:
            training = "allen Ideen"
        else:
            training = dataset + " Ideen"

        if "Dups" in key:
            dups = " mit Duplikat-Filterung"
        else:
            dups = ""

        plt.plot(compreclist, "b", label="Präzision 5-Wort Bayes")
        plt.plot(comreclist, "g", label="Trefferquote 5-Wort Bayes")
        plt.plot(baypreclist, "y", label="Präzision 1-Wort Bayes")
        plt.plot(bayreclist, "r", label="Trefferquote 1-Wort Bayes")
        plt.title(('Bayes Klassifikatoren für ' + dataset + "trainiert auf " + training + dups))
        plt.xlabel('Spam-Wahrscheinlichkeit der Idee')
        plt.legend(loc="best")
        plt.show()

# y_test, y_score, challenge, dups
def plot_lin_Classifier():
    resultdict = import_results()
    fpr_fabric = dict()
    tpr_fabric = dict()
    roc_auc_fabric = dict()
    fpr_bionic = dict()
    tpr_bionic = dict()
    roc_auc_bionic = dict()
    fpr_tco = dict()
    tpr_tco = dict()
    roc_auc_tco = dict()
    for key in resultdict.keys():
        if key == "fabricresults":
            linclasspred = [x for x in resultdict[key]["complexbayes"]]
#            linclassprob = [x[1] for x in resultdict[key]["bayes"]]
#            for i in range(len(linclasspred)):
#                if not linclasspred[i]:
#                    linclassprob[i] = 1 - linclassprob[i]
            for i in [0, 1]:
                fpr_fabric[i], tpr_fabric[i], _ = roc_curve(resultdict[key]["actual"], linclasspred)
                roc_auc_fabric[i] = auc(fpr_fabric[i], tpr_fabric[i])
        elif key == "tcoresults":
            linclasspred = [x for x in resultdict[key]["complexbayes"]]
#            linclassprob = [x[1] for x in resultdict[key]["bayes"]]
#            for i in range(len(linclasspred)):
#                if not linclasspred[i]:
#                    linclassprob[i] = 1 - linclassprob[i]
            for i in [0, 1]:
                fpr_tco[i], tpr_tco[i], _ = roc_curve(resultdict[key]["actual"], linclasspred)
                roc_auc_tco[i] = auc(fpr_tco[i], tpr_tco[i])
        elif key == "bionicresults":
            linclasspred = [x for x in resultdict[key]["complexbayes"]]
#            linclassprob = [x[1] for x in resultdict[key]["bayes"]]
#            for i in range(len(linclasspred)):
#                if not linclasspred[i]:
#                    linclassprob[i] = 1 - linclassprob[i]
            for i in [0, 1]:
                fpr_bionic[i], tpr_bionic[i], _ = roc_curve(resultdict[key]["actual"], linclasspred)
                roc_auc_bionic[i] = auc(fpr_bionic[i], tpr_bionic[i])

#    fpr = dict()
#    tpr = dict()
#    roc_auc = dict()
#    for i in [0, 1]:
#        fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
#        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    plt.plot(fpr_tco[1], tpr_tco[1], color="darkorange", lw=lw, label="TCO" % roc_auc_tco[1] + " (area = %0.2f)" % roc_auc_tco[1])
    plt.plot(fpr_bionic[1], tpr_bionic[1], color="navy", lw=lw, label="Bionic Radar" % roc_auc_bionic[1] + " (area = %0.2f)" % roc_auc_bionic[1])
    plt.plot(fpr_fabric[1], tpr_fabric[1], color="green", lw=lw, label="Fabric Display" % roc_auc_fabric[1] + " (area = %0.2f)" % roc_auc_fabric[1])
    plt.plot([0, 1], [0, 1], color="cornflowerblue", lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
#    plt.title(challenge)
    plt.legend(loc="lower right")
#    if dups:
    plt.savefig("Data/Plots/combayesTestres.png")
#    else:
#        plt.savefig(variables.firstrunpath + variables.plotpath + "Training/ROC_linClass_" + challenge + ".png")


def import_results():
    fabricresults = evaluationHelper.convertResults(list(
        importDataHelper.readcsvdata(variables.firstrunpath + variables.testresultpath + "first/fabricDisplay.csv"))[0])
    fabricresultsNoDups = evaluationHelper.convertResults(list(
        importDataHelper.readcsvdata(variables.firstrunpath + variables.testresultpath + "first/evaluationResultsNoDupsfabricDisplay.csv"))[0])
    fabricresultsNoDupsAll = evaluationHelper.convertResults(list(
        importDataHelper.readcsvdata(variables.firstrunpath + variables.testresultpath + "first/evaluationResultsNoDupsAllfabricDisplay.csv"))[0])
    fabricresultsAll = evaluationHelper.convertResults(list(
        importDataHelper.readcsvdata(variables.firstrunpath + variables.testresultpath + "first/evaluationAllfabricDisplay.csv"))[0])

    tcoresults = evaluationHelper.convertResults(list(
        importDataHelper.readcsvdata(variables.firstrunpath + variables.testresultpath + "first/TCO.csv"))[0])
    tcoresultsNoDups = evaluationHelper.convertResults(list(
        importDataHelper.readcsvdata(variables.firstrunpath + variables.testresultpath + "first/evaluationResultsNoDupsTCO.csv"))[0])
    tcoresultsNoDupsAll = evaluationHelper.convertResults(list(
        importDataHelper.readcsvdata(variables.firstrunpath + variables.testresultpath + "first/evaluationResultsNoDupsAllTCO.csv"))[0])
    tcoresultsAll = evaluationHelper.convertResults(list(
        importDataHelper.readcsvdata(variables.firstrunpath + variables.testresultpath + "first/evaluationAllTCO.csv"))[0])

    bionicresults = evaluationHelper.convertResults(list(
        importDataHelper.readcsvdata(variables.firstrunpath + variables.testresultpath + "first/bionicRadar.csv"))[0])
    bionicresultsNoDups = evaluationHelper.convertResults(list(
        importDataHelper.readcsvdata(variables.firstrunpath + variables.testresultpath + "first/evaluationResultsNoDupsbionicRadar.csv"))[0])
    bionicresultsNoDupsAll = evaluationHelper.convertResults(list(
        importDataHelper.readcsvdata(variables.firstrunpath + variables.testresultpath + "first/evaluationResultsNoDupsAllbionicRadar.csv"))[0])
    bionicresultsAll = evaluationHelper.convertResults(list(
        importDataHelper.readcsvdata(variables.firstrunpath + variables.testresultpath + "first/evaluationAllbionicRadar.csv"))[0])

    return {"fabricresults": fabricresults, "fabricresultsNoDups": fabricresultsNoDups,
                  "fabricresults All": fabricresultsAll, "fabricresults AllNoDups": fabricresultsNoDupsAll,
                  "tcoresults": tcoresults, "tcoresultsNoDups": tcoresultsNoDups,
                  "tcoresults All": tcoresultsAll, "tcoresults AllNoDups": tcoresultsNoDupsAll,
                  "bionicresults": bionicresults, "bionicresultsNoDups": bionicresultsNoDups,
                  "bionicresults All": bionicresultsAll, "bionicresults AllNoDups": bionicresultsNoDupsAll}
