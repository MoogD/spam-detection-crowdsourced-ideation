from Helper import importDataHelper
from Filter import duplicateDetection, bayes, complexBayes, linearClassifier, USEClassifier
from Evaluation import  evaluateSystem
from Visualization import confusionMatrix
import spamFilter
import variables

# Variables for filters:

charcount = 15
wordcount = 2
challenges = ["TCO", "bionicRadar", "fabricDisplay"]

def main():
    evaluateSystem.eval_USE()
 #   evaluateSystem.eval_newAll()

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
        importDataHelper.writecsvfile(variables.resultpath + key + "unreviewed.csv", idealistchallenge[key][0].keys(), idealistchallenge[key])

if __name__ == '__main__':
    main()
