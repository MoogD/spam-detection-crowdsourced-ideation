from os import listdir
from os.path import isfile, join
import time

import nltk
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import brown

import spamFilter
from Helper import importDataHelper
from Filter import duplicateDetection, bayes, complexBayes
import variables
from Visualization import confusionMatrix


def commandline_control():
    dataset = None
    while(True):
        print("Select Task by entering the number: ")
        print("1: Filter Duplicates")
        print("2: Train single word Bayes")
        print("3: Train 5-Word Bayes")
        print("4: Classify Ideaset")
        dataset = taskswitcher(int(input("What do you want to do? ")), dataset)


def taskswitcher(task, dataset):
    tasks = {1: duplicatefilter, 2: trainbayes, 3: traincomplexbayes, 4: classifyideas}
    func = tasks.get(task, commandline_control)
    return func(dataset)


def duplicatefilter(dataset=None):
    if dataset is None:
        print("Select a dataset: ")
        i = 0
        print("Classified datasets")
        filesclass = []
        for file in listdir(variables.importpathclassified):
            if isfile(join(variables.importpathclassified, file)):
                print("", i, ": ", file)
                filesclass.append((variables.importpathclassified, file))
                i += 1
        print("Unclassified datasets")
        for file in listdir(variables.importpathunclassified):
            if isfile(join(variables.importpathunclassified, file)):
                print("", i, ": ", file)
                filesclass.append((variables.importpathunclassified, file))
                i += 1
        selected = int(input("Which dataset do you want to use? "))
        path = filesclass[selected][0]
        filename, fileformat = filesclass[selected][1].replace(".", ' ').split()
        if 'csv' in fileformat:
            idealist = list(importDataHelper.readcsvdata(join(path, filename + '.' + fileformat)))
        else:
            idealist = list(importDataHelper.readxmldata(join(path, filename + '.' + fileformat)))
    else:
        fileformat = dataset[3]
        filename = dataset[2]
        path = dataset[1]
        idealist = dataset[0]
    idealist = duplicateDetection.filterduplikates(idealist, variables.duplicateresultpath + filename + 'Duplicates.csv')
    return idealist, path, filename, fileformat

def trainbayes(dataset=None):
    if dataset is None:
        print("Select a dataset: ")
        i = 0
        print("Classified datasets")
        filesclass = []
        for file in listdir(variables.importpathclassified):
            if isfile(join(variables.importpathclassified, file)):
                print("", i, ": ", file)
                filesclass.append((variables.importpathclassified, file))
                i += 1
        print("Unclassified datasets")
        for file in listdir(variables.importpathunclassified):
            if isfile(join(variables.importpathunclassified, file)):
                print("", i, ": ", file)
                filesclass.append((variables.importpathunclassified, file))
                i += 1
        selected = int(input("Which dataset do you want to use? "))
        path = filesclass[selected][0]
        filename, fileformat = filesclass[selected][1].replace(".", ' ').split()
        if 'csv' in fileformat:
            idealist = list(importDataHelper.readcsvdata(join(path, filename + '.' + fileformat)))
        else:
            idealist = list(importDataHelper.readxmldata(join(path, filename + '.' + fileformat)))
    else:
        idealist = dataset[0]
    delete = ""
    while('y' not in delete or 'n' in delete):
        delete = input("Do you want to override old bayes results (y/n): ").lower()
    start = time.process_time_ns()
    if 'y' in delete:
        spamdict = {}
        hamdict = {}
    else:
        spamdict = bayes.getspamtokens()  # load data this time to get data from both datasets
        hamdict = bayes.gethamtokens() # load data this time to get data from both datasets
    bayes.trainbayes(idealist, spamdict, hamdict)
    duration = time.process_time_ns() - start
    print("Duration bayestraining: ", duration / 1000000000, "seconds")
    return None


def traincomplexbayes(dataset=None):
    if dataset is None:
        print("Select a dataset: ")
        i = 0
        print("Classified datasets")
        filesclass = []
        for file in listdir(variables.importpathclassified):
            if isfile(join(variables.importpathclassified, file)):
                print("", i, ": ", file)
                filesclass.append((variables.importpathclassified, file))
                i += 1
        print("Unclassified datasets")
        for file in listdir(variables.importpathunclassified):
            if isfile(join(variables.importpathunclassified, file)):
                print("", i, ": ", file)
                filesclass.append((variables.importpathunclassified, file))
                i += 1
        selected = int(input("Which dataset do you want to use? "))
        path = filesclass[selected][0]
        filename, fileformat = filesclass[selected][1].replace(".", ' ').split()
        if 'csv' in fileformat:
            idealist = list(importDataHelper.readcsvdata(join(path, filename + '.' + fileformat)))
        else:
            idealist = list(importDataHelper.readxmldata(join(path, filename + '.' + fileformat)))
    else:
        idealist = dataset[0]
    delete = ""
    while('y' not in delete or 'n' in delete):
        delete = input("Do you want to override old 5-word bayes results (y/n): ").lower()
    start = time.process_time_ns()
    if 'y' in delete:
        spamdictcom = {}
        hamdictcom = {}
    else:
        spamdictcom = complexBayes.getspamtokens() # load data this time to get data from both datasets
        hamdictcom = complexBayes.gethamtokens() # load data this time to get data from both datasets
    complexBayes.trainbayes(idealist, spamdictcom, hamdictcom)
    duration = time.process_time_ns() - start
    print("Duration (complex) bayestraining: ", duration / 1000000000, "seconds")
    return None


def classifyideas(dataset=None):
    if dataset is None:
        print("Select a dataset: ")
        i = 0
        print("Classified datasets")
        filesclass = []
        for file in listdir(variables.importpathclassified):
            if isfile(join(variables.importpathclassified, file)):
                print("", i, ": ", file)
                filesclass.append((variables.importpathclassified, file))
                i += 1
        print("Unclassified datasets")
        for file in listdir(variables.importpathunclassified):
            if isfile(join(variables.importpathunclassified, file)):
                print("", i, ": ", file)
                filesclass.append((variables.importpathunclassified, file))
                i += 1
        selected = int(input("Which dataset do you want to use? "))
        path = filesclass[selected][0]
        filename, fileformat = filesclass[selected][1].replace(".", ' ').split()
        if 'csv' in fileformat:
            idealist = list(importDataHelper.readcsvdata(join(path, filename + '.' + fileformat)))
        else:
            idealist = list(importDataHelper.readxmldata(join(path, filename + '.' + fileformat)))
    else:
        fileformat = dataset[3]
        filename = dataset[2]
        path = dataset[1]
        idealist = dataset[0]
    bayesbool = 'y' in input("Do you want to use single word bayes to classify? (y/n) ").lower()
    complbayesbool = 'y' in input("Do you want to use 5-word bayes to classify? (y/n) ").lower()
    filtersystembool = 'y' in input("Do you want to use the Filtersystem to classify? (y/n) ").lower()
    if bayesbool:
        wordprobs = bayes.gettokenprobs()
    if complbayesbool:
        wordprobscom = complexBayes.gettokenprobs()
    if filtersystembool:
        unigram_tagger, st = prepare_tagger()

    spamlist = []
    applied_filters = {}
    pred = []
    actual = []
    fplist = []
    fnlist = []
    start1 = time.time()

    for row in idealist:
        row['TRIGGERED'] = []
        row['PREDICTION'] = "Ham"
        if bayesbool:
            bayesprob = bayes.classify(row['DESCRIPTION'], wordprobs)
            if bayesprob > 0.8:
                row['TRIGGERED'].append("bayes")
                applied_filters["bayes"] = int(applied_filters.get("bayes", 0)) + 1
                row['PREDICTION'] = "Spam"
        if complbayesbool:
            combayesprob = complexBayes.classify(row['DESCRIPTION'], wordprobscom)
            if combayesprob > 0.8:
                row['TRIGGERED'].append("complex bayes: " + str(combayesprob))
                applied_filters["complex bayes"] = int(applied_filters.get("complex bayes", 0)) + 1
                row['PREDICTION'] = "Spam"
        if filtersystembool:
            row = spamFilter.classifyidea(row, unigram_tagger, st)
        actual.append("spam" in row.get('SPAM', "") or "unusable" in row.get("STATUS", ""))
        pred.append(row['PREDICTION'] == "Spam")
        for filter in row['TRIGGERED']:
            if 'bayes' not in filter:
                applied_filters[filter] = int(applied_filters.get(filter, 0)) + 1
        spamlist.append(row)
        if row['PREDICTION'] == "Spam" and ("ham" in row.get('SPAM', "") or row.get("STATUS", "") == "usable"):
            fplist.append(row)
        elif row['PREDICTION'] == "Ham" and ("spam" in row.get('SPAM', "") or "unusable" in row.get("STATUS", "")):
            fnlist.append(row)
    cm = confusionMatrix.create_confusionmatrix(actual, pred)
    confusionMatrix.print_confusionmatrix(cm, True)
    description = "just filtersystem, Test enumeration fix with iui dataset"

    confusionMatrix.save_confusionmatrix(cm, variables.resultpath + "ConfusionMatrices.csv", applied_filters,
                                         description, filename)
    duration1 = time.time() - start1
    print("Duration1: ", duration1, "seconds")
    print(applied_filters)

    ###################### Save results ######################
    #    importDataHelper.writecsvfile(variables.resultpath + 'IdeaDataSpam2.csv', spamlist[0].keys(), spamlist)
    if len(fplist) > 0:
        importDataHelper.writecsvfile(variables.filterresults + filename + '_fp.csv', fplist[0].keys(), fplist)
    if len(fnlist) > 0:
        importDataHelper.writecsvfile(variables.filterresults + filename + '_fn.csv', fnlist[0].keys(), fnlist)
    return None




def prepare_tagger():
    start = time.process_time_ns()
    ### prepare unigram tagger with brown corpus
    brown_tagged_sents = brown.tagged_sents(
        categories=['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned',
                    'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction'])
    unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
    ## prepare name tagger
    st = StanfordNERTagger('stanford-ner-2018-10-16/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz', 'stanford-ner-2018-10-16/stanford-ner-2018-10-16/stanford-ner.jar', encoding='utf-8')
    duration = time.process_time_ns() - start
    print("Duration preparing tagger: ", duration / 1000000000, "seconds")
    return (unigram_tagger, st)