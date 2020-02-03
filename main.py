import pickle

from Helper import importDataHelper
from Filter import duplicateDetection, bayes, complexBayes, linearClassifier, USEClassifier
from Evaluation import evaluateSystem
from Visualization import confusionMatrix
import spamFilter
import variables

import xml.etree.ElementTree as ET
import argparse
import pandas as pd
import os
# Variables for filters:

charcount = 15
wordcount = 2
challenges = ["TCO", "bionicRadar", "fabricDisplay"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to a csv or xml file with ideas")

    parser.add_argument("-t", "--train", help="to train the system. Requires classified ideas.", action="store_true")
    parser.add_argument("--challenge", help="give a challenge to use instead of the challenges given in an idea")
    args = parser.parse_args()
    filename, fileformat = os.path.basename(args.path).split('.')
    if fileformat == 'csv':
        idealist = list(importDataHelper.readcsvdata(args.path))
    elif fileformat == 'xml':
        idealist = importDataHelper.readxmldata(args.path)
    else:
        print("Can not read the file, please use csv or xml files")
        return 1
    challengelists = {}
    # Divide idea in challenges or use the given challenge
    if args.challenge is None:
        for idea in idealist:
            challenge = idea.get("CHALLENGE", "Cross-Domain")
            challengelists[challenge] = challengelists.get(challenge, [])
            challengelists[challenge].append(idea)
    else:
        challengelists[args.challenge] = idealist
    if args.train:
        for elem in challengelists:
            train(challengelists[elem], elem)
    else:
        classifiedlist = []
        for elem in challengelists:
            if fileformat == "csv":
                classifiedlist += classify(challengelists[elem], elem, fileformat)
                importDataHelper.writecsvfile(os.path.dirname(args.path) + "/" + filename + "_classified.csv", classifiedlist[0].keys(), classifiedlist)
            else:
                idealist = classify(idealist, elem, fileformat)
                idealist.write(os.path.dirname(args.path) + "/" + filename + "_classified.xml")




def train(idealist, challenge=None):
    if not challenge is None:
        print("Train the Bayes classificators...")
        bayes.trainbayes(idealist, challenge=challenge, delete=False)
        complexBayes.trainbayes(idealist, challenge=challenge, delete=False)
        print("Bayes classificators trained!")
        print("preparing rulebased Filtersystem...")
        unigram_tagger, st = spamFilter.prepare_tagger()
        print("Rulebased Filtersystem prepared!")
        data = {"DESCRIPTION": [], "Spam": []}
        features = {}
        print("Classifying Ideas with Filtersystem...")
        for idea in idealist:
            idea['TRIGGERED'] = []
            idea, feature = spamFilter.classify_and_get_idea(idea, unigram_tagger, st)
            data["DESCRIPTION"].append(idea["DESCRIPTION"])
            for ideakey in feature.keys():
                if feature[ideakey] == 1:
                    features[ideakey] = features.get(ideakey, [])
                    features[ideakey].append(1)
                else:
                    features[ideakey] = features.get(ideakey, [])
                    features[ideakey].append(0)
                feature[ideakey] = [feature[ideakey]]
            if idea["STATUS"] == "unusable":
                features["Spam"] = features.get("Spam", [])
                features["Spam"].append(1)
                data["Spam"].append(1)
            elif idea["STATUS"] == "usable":
                features["Spam"] = features.get("Spam", [])
                features["Spam"].append(0)
                data["Spam"].append(0)
            elif "spam" in idea.get("SPAM", ""):
                features["Spam"] = features.get("Spam", [])
                features["Spam"].append(1)
                data["Spam"].append(1)
            else:
                features["Spam"] = features.get("Spam", [])
                features["Spam"].append(0)
                data["Spam"].append(0)
        print("Ideas classified by Filtersystem!")
        linClass, coef = linearClassifier.train_linear_classifier(features)
######## Test classify function
        # wordprobs = bayes.gettokenprobs(challenge=challenge)
        # comwordprobs = complexBayes.gettokenprobs(challenge=challenge)
########
        path = variables.classpath + "/" + challenge
        if os.path.exists(variables.classpath + "/" + challenge):
            with open((variables.classpath + "/" + challenge + "/linClassAttributes.txt"), "wb") as fp:  # Pickling
                pickle.dump(linClass, fp)
            useest = USEClassifier.train_classifier_idealist(pd.DataFrame(data), path)
        else:
            try:
                os.mkdir(variables.classpath + "/" + challenge)
                path = variables.classpath + "/" + challenge
                useest = USEClassifier.train_classifier_idealist(pd.DataFrame(data), path)
            except OSError:
                print("Creation of the directory %s failed" % (variables.classpath + challenge))
            # if path is not None:
            #     with open((variables.classpath + "/" + challenge + "/linClassAttributes.txt"), "wb") as fp:  # Pickling
            #         pickle.dump(linClassdict, fp)
######## Test classify function
        # predbay = []
        # predcombay = []
        # predUSE = []
        # predUSEprob = []
        # predLin = []
        # predLinprob = []
        # for idea in idealist:
        #     idea['TRIGGERED'] = idea.get("TRIGGERED", [])
        #     predbay.append(bayes.classify(idea["DESCRIPTION"], wordprobs))
        #     predcombay.append(complexBayes.classify(idea["DESCRIPTION"], comwordprobs))
        #     # data = {}
        #     # data["DESCRIPTION"].append(idea["DESCRIPTION"])
        #     # if "unusable" in idea.get("STATUS", ""):
        #     #     data["SPAM"].append(1)
        #     # elif "usable" in idea.get("STATUS", ""):
        #     #     data["SPAM"].append(0)
        #     # elif "spam" in idea.get("SPAM", ""):
        #     #     data["SPAM"].append(1)
        #     # else:
        #     #     data["SPAM"].append(0)
        #     x = USEClassifier.classify(useest, {"DESCRIPTION": idea["DESCRIPTION"]})
        #     predUSE.append(x[0])
        #     predUSEprob.append(x[1])
        #     # idea, ideadata = spamFilter.classify_and_get_idea(idea, unigram_tagger, st)
        #     # features = {}
        #     # for ideakey in ideadata.keys():
        #     #     if ideadata[ideakey] == 1:
        #     #         features[ideakey] = features.get(ideakey, [])
        #     #         features[ideakey].append(1)
        #     #     else:
        #     #         features[ideakey] = features.get(ideakey, [])
        #     #         features[ideakey].append(0)
        #     #     ideadata[ideakey] = [ideadata[ideakey]]
        #     # x = linearClassifier.classify(pd.DataFrame(ideadata), linClass)
        #     # predLin.append(x[0])
        #     # predLinprob.append(x[1])
        # return predbay, predcombay, predUSE, predUSEprob, predLin, predLinprob


def classify(idealist, challenge=None, type=None):
    print(type)
    if challenge is not None:
        wordprobs = bayes.gettokenprobs(challenge=challenge)
        comwordprobs = complexBayes.gettokenprobs(challenge=challenge)
        path = variables.classpath + "/" + challenge
        est = USEClassifier.load_estimator(path)
        with open((variables.classpath + "/" + challenge + "/linClassAttributes.txt"), "rb") as fp:  # Pickling
            linClass = pickle.load(fp)
        unigram_tagger, st = spamFilter.prepare_tagger()
        if type == "xml":
            ideas = idealist.getroot()
        else:
            ideas = idealist
        for idea in ideas:
            if type == "xml":
                for att in idea:
                    if att.tag == "{http://purl.org/gi2mo/ns#}content":
                        description = att.text
                        triggered = []
            else:
                triggered = idea.get("TRIGGERED", [])
                description = idea["DESCRIPTION"]
            prediction = 0.0
            bay = bayes.classify(description, wordprobs)
            combay = complexBayes.classify(description, comwordprobs)
            classcount = []
            if (bay >= 0.9):
                triggered.append("1-WordBayes: {}".format(bay))
                classcount.append(bay)
            if combay >= 0.9:
                triggered.append("5-WordBayes: {}".format(combay))
                classcount.append(combay)
            use = USEClassifier.classify(est, {"DESCRIPTION": description})
            if (use[0]== 1):
                triggered.append("SentenceEmbedding: {}".format(use[1]))
                classcount.append(use[1])
            merke, filter, ideadata = spamFilter.classify_and_get_idea(description, unigram_tagger, st)
            features = {}
            for ideakey in ideadata.keys():
                if ideadata[ideakey] == 1:
                    features[ideakey] = features.get(ideakey, [])
                    features[ideakey].append(1)
                else:
                    features[ideakey] = features.get(ideakey, [])
                    features[ideakey].append(0)
                ideadata[ideakey] = [ideadata[ideakey]]
            lin = linearClassifier.classify(pd.DataFrame(ideadata), linClass)
            if (lin[0] == 1):
                triggered.append("linearClassifier: {}".format(lin[1]))
                classcount.append(lin[1])
            triggered += filter
            if len(classcount) > 1:
                prediction = sum(classcount)/len(classcount)
            if type == "xml":
                ET.SubElement(idea, "Spamsystem", {"Spamprob": str(round(prediction, 2)),"Triggered": triggered})
            else:
                idea["SPAMPROB"] = round(prediction, 2)
                idea["TRIGGERED"] = triggered
        if type == "xml":
            return idealist
        else:
            return ideas






def test():
#    idealist = list(importDataHelper.readxmldata(variables.importpathunclassified + 'IdeaData.xml'))
    idealist = list(importDataHelper.readcsvdata(variables.importpathclassified + "ideas-with-challenges.csv"))
    idealistchallenge = {"bionicRadar": [], "fabricDisplay": []}
    print(len(idealist))
    i = 0
    j = 0
    k = 0
    for idea in idealist:
        if idea["STATUS"] == "unreviewed":
            if "bionic" in idea["CHALLENGE"].lower():
                i += 1
                idealistchallenge["bionicRadar"].append(idea)
            elif "fabric" in idea["CHALLENGE"].lower():
                j += 1
                idealistchallenge["fabricDisplay"].append(idea)
            else:
                k += 1
    print("unreviewed bionic: ", i)
    print("unreviewed fabric: ", j)
    print("unreviewed others: ", k)

    idealisttrainingschallenge = {}
    idealisttrainingschallenge["fabricDisplay"] = list(importDataHelper.readcsvdata(variables.ideadbpath + 'fabricDisplay.csv'))
    idealisttrainingschallenge["bionicRadar"] = list(importDataHelper.readcsvdata(variables.ideadbpath + 'bionicRadar.csv'))
    idealisttrainingschallenge["TCO"] = list(importDataHelper.readcsvdata(variables.ideadbpath + 'TCO.csv'))

    idealisttrainingschallengewodups = {}
    idealisttrainingschallengewodups["fabricDisplay"] = list(importDataHelper.readcsvdata(variables.ideadbwithoutduppath + "fabricDisplay.csv"))
    idealisttrainingschallengewodups["bionicRadar"] = list(importDataHelper.readcsvdata(variables.ideadbwithoutduppath + "bionicRadar.csv"))
    idealisttrainingschallengewodups["TCO"] = list(importDataHelper.readcsvdata(variables.ideadbwithoutduppath + "TCO.csv"))

    idealistmixedtraining = idealisttrainingschallenge["fabricDisplay"] + idealisttrainingschallenge["bionicRadar"] + idealisttrainingschallenge["TCO"]
    idealistmixedtrainingwithoutdups = idealisttrainingschallengewodups["fabricDisplay"] + idealisttrainingschallengewodups["bionicRadar"] + idealisttrainingschallengewodups["TCO"]

    for key in idealistchallenge.keys():
        idealisttraining = idealisttrainingschallenge[key]
        idealisttrainingwithoutdups = list(importDataHelper.readcsvdata(variables.ideadbwithoutduppath + key + ".csv"))



#        idealistchallengewithoutdups = duplicateDetection.filterduplikates(idealistchallenge[key], variables.resultpath + "test3.csv", idealisttrainingwithoutdups)
        print("duplicate detection done")

        bayes.trainbayes(idealisttraining, challenge=key, delete=True)
        bayes.trainbayes(idealisttrainingwithoutdups, challenge=key, delete=True, duplicates=True)
        print("bayes training TCO complete")

        bayes.trainbayes(idealistmixedtraining, delete=True)
        bayes.trainbayes(idealistmixedtrainingwithoutdups,  delete=True, duplicates=True)
        print("bayes training mixed complete")

        wordprobs = bayes.gettokenprobs(challenge=key)
        wordprobswithoutdups = bayes.gettokenprobs(challenge=key, duplicates=True)

        wordprobsmixed = bayes.gettokenprobs()
        wordprobsmixedwithoutdups = bayes.gettokenprobs(duplicates=True)
        print("loaded probs")
        complexBayes.trainbayes(idealisttraining, challenge=key, delete=True)
        complexBayes.trainbayes(idealisttrainingwithoutdups, challenge=key, delete=True, duplicates=True)
        print("complex bayes training TCO complete")

        complexBayes.trainbayes(idealistmixedtraining, delete=True)
        complexBayes.trainbayes(idealistmixedtrainingwithoutdups,  delete=True, duplicates=True)
        print("complex bayes training mixed complete")

        comwordprobs = complexBayes.gettokenprobs(challenge=key)
        comwordprobswithoutdups = complexBayes.gettokenprobs(challenge=key, duplicates=True)

        comwordprobsmixed = complexBayes.gettokenprobs()
        comwordprobsmixedwithoutdups = complexBayes.gettokenprobs(duplicates=True)
        print("loaded probs complex")

        linclass, lincoeff = linearClassifier.train_linear_classificator(key)
        print(lincoeff)
        linclassmixed, lincoeffmixed = linearClassifier.train_linear_classificator("all")
        print(lincoeffmixed)

        useest = USEClassifier.train_classifier(key)
        useestmixed = USEClassifier.train_classifier("all")
        print("trained USE")

        unigram_tagger, st = spamFilter.prepare_tagger()

        i = 1
        for idea in idealistchallenge[key]:
            print (i)
            idea["TRIGGERED"] = [""]
            # classify with challenge bayes with duplicates
            bayesprob = bayes.classify(idea["DESCRIPTION"], wordprobs)
            # classify with challenge bayes without duplicates
            bayesprobdup = bayes.classify(idea["DESCRIPTION"], wordprobswithoutdups)
            # classify with mixed challenge bayes with duplicates
            bayesprobmixed = bayes.classify(idea["DESCRIPTION"], wordprobsmixed)
            # classify with mixed challenge bayes without duplicates
            bayesprobmixedwithoutdup = bayes.classify(idea["DESCRIPTION"], wordprobsmixedwithoutdups)

            combayesprob = complexBayes.classify(idea["DESCRIPTION"], comwordprobs)
            # classify with challenge bayes without duplicates
            combayesprobdup = complexBayes.classify(idea["DESCRIPTION"], comwordprobswithoutdups)
            # classify with mixed challenge bayes with duplicates
            combayesprobmixed = complexBayes.classify(idea["DESCRIPTION"], comwordprobsmixed)
            # classify with mixed challenge bayes without duplicates
            combayesprobmixedwithoutdup = complexBayes.classify(idea["DESCRIPTION"], comwordprobsmixedwithoutdups)

            # classify with challenge USE:
            useclass, useclassprob = USEClassifier.classify(useest, idea)
            # classify with mixed challenge USE:
            usemixedclass, usemixedclassprob = USEClassifier.classify(useestmixed, idea)

            idea, ideadata = spamFilter.classify_and_get_idea(idea, unigram_tagger, st)
            allnull = True
            for keytest in ideadata.keys():
                ideadata[keytest] = [ideadata[keytest]]
                if ideadata[keytest] == 1:
                    allnull = False
            if not allnull:
                linclasspred, linclassprob = linearClassifier.classify(ideadata, linclass)
                linmixedclasspred, linmixedclassprob = linearClassifier.classify(ideadata, linclassmixed)
            else:
                linclasspred, linclassprob = 0, 0
                linmixedclasspred, linmixedclassprob = 0, 0
            idea["PREDICTION"] = "Bayes: " + str(bayesprobdup) + ", complexBayes " + str(combayesprobdup) + ", linClass: " + str(linmixedclasspred) + " " + str(linmixedclassprob) + ", USE: " + str(useclass) + " " + str(useclassprob)
            idea["OTHERBayes"] = "BayesTCO: " + str(bayesprob) + ", BayesMixed " + str(bayesprobmixed) + ", BayesMixed w/o dups " + str(bayesprobmixedwithoutdup) + ", compl BayesTCO: " + str(combayesprob) + ", compl BayesMixed: " + str(combayesprobmixed) + ", compl BayesMixed w/o dups: " + str(combayesprobmixedwithoutdup)
            idea["OTHERS"] = "Lin Class: " + str(linclasspred) + " " + str(linclassprob) + ", USE mixed: " + str(usemixedclass) + " " + str(usemixedclassprob)

            i += 1
        importDataHelper.writecsvfile(variables.resultpath + key + "unreviewed.csv",
                                      idealistchallenge[key][0].keys(), idealistchallenge[key])

if __name__ == '__main__':
    main()
