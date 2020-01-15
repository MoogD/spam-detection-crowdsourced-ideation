import variables
from Helper import importDataHelper


def classify_unreviewed():
    idealist = list(importDataHelper.readcsvdata("Data/Results/fabricDisplayunreviewed.csv"))
    idealist2 = list(importDataHelper.readcsvdata("Data/Results/fabricDisplayClassified.csv"))
    print("bionic Radar:")
    for idea in idealist:
        if idea["ID"] in [ideas["ID"] for ideas in idealist2]:
            idealist.remove(idea)
    print(len(idealist))
    for idea in idealist:
        print(" ")
        if "usable" not in idea.get("STATUS", ""):
            print("Content: " + idea["DESCRIPTION"])
            print("Prediction: " + idea["PREDICTION"])
            print("Bayes: " + idea["OTHERBayes"])
            print("Others: " + idea["OTHERS"])
            print("Filter: " + idea["TRIGGERED"])
            x = input("Spam? (y/n)")
            if 'y' in x:
                idea["STATUS"] = "unusable"
                idealist2.append(idea)
                idealist.remove(idea)
            elif 'n' in x:
                idea["STATUS"] = "usable"
                idealist2.append(idea)
                idealist.remove(idea)
            else:
                importDataHelper.writecsvfile("Data/Results/fabricDisplayClassified.csv",
                                              idealist2[0].keys(), idealist2)
                importDataHelper.writecsvfile("Data/Results/fabricDisplayunreviewed.csv",
                                              idealist[0].keys(), idealist)
    importDataHelper.writecsvfile("Data/Results/fabricDisplayClassified.csv",
                                  idealist2[0].keys(), idealist2)
    importDataHelper.writecsvfile("Data/Results/fabricDisplayunreviewed.csv",
                                  idealist[0].keys(), idealist)
