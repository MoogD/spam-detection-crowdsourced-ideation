from os import listdir
from os.path import isfile, join
import time

import nltk
from nltk.corpus import brown
from nltk.tag.stanford import StanfordNERTagger

import variables
from Filter import bayes
from Filter import complexBayes
from Filter import textContentFilter
from Filter import textDataFilter
from Helper import importDataHelper
from Visualization import confusionMatrix

countless = [5, 10, 15, 20, 25, 30]
countmore = [50, 100, 150, 200, 250]
countwords = [2, 5, 8,  10, 15]

def evaluate_filtersystem():
    resultlist = []
    unigram, st = prepare_tagger()
    for file in listdir(variables.importpathclassified):
        if isfile(join(variables.importpathclassified, file)):
            if ".csv" in file:
                idealist = list(importDataHelper.readcsvdata(join(variables.importpathclassified, file)))
            elif ".xml" in file:
                idealist = list(importDataHelper.readxmldata(join(variables.importpathclassified, file)))
            else:
                print("Not able to read all files (just csv and xml are supported)")
                return 1
            for filter in textDataFilter.textDataFilterList:
                if "count" in str(filter):
                    if "more" in filter.__name__:
                        for count in countmore:
                            cm = evaluate_filter(filter, idealist, count)
                            result = {"Dataset": file, "Filter": filter.__name__, "Variable": count}
                            if cm is not None:
                                result.update(cm.stats())
                            resultlist.append(result)
                    elif "less" in filter.__name__:
                        for count in countless:
                            cm = evaluate_filter(filter, idealist, count)
                            result = {"Dataset": file, "Filter": filter.__name__, "Variable": count}
                            if cm is not None:
                                result.update(cm.stats())
                            resultlist.append(result)
                    elif "word" in filter.__name__:
                        for count in countwords:
                            cm = evaluate_filter(filter, idealist, count)
                            result = {"Dataset": file, "Filter": filter.__name__, "Variable": count}
                            if cm is not None:
                                result.update(cm.stats())
                            resultlist.append(result)
                else:
                    cm = evaluate_filter(filter, idealist)
                    result = {"Dataset": file, "Filter": filter.__name__, "Variable": "None"}
                    if cm is not None:
                        result.update(cm.stats())
                    resultlist.append(result)
            for filter in textContentFilter.textContentFilterlist:
                if "unigram" in filter.__name__:
                    cm = evaluate_filter(filter, idealist, unigram)
                    result = {"Dataset": file, "Filter": filter.__name__, "Variable": "UnigramTagger"}
                    if cm is not None:
                        result.update(cm.stats())
                    resultlist.append(result)
                elif "containsnames" in filter.__name__:
                    cm = evaluate_filter(filter, idealist, st)
                    result = {"Dataset": file, "Filter": filter.__name__, "Variable": "StanfordNERTagger"}
                    if cm is not None:
                        result.update(cm.stats())
                    resultlist.append(result)
                else:
                    cm = evaluate_filter(filter, idealist)
                    result = {"Dataset": file, "Filter": filter.__name__, "Variable": "None"}
                    if cm is not None:
                        result.update(cm.stats())
                    resultlist.append(result)
                print(filter.__name__)
    importDataHelper.writecsvfile(variables.resultpath + "FilterEvaluation.csv", resultlist[0].keys(), resultlist)



def evaluate_filter(filter, idealist, var=None):
    actual = []
    pred = []
    for row in idealist:
        if var is not None:
            pred.append(filter(row["DESCRIPTION"], var))
        else:
            pred.append(filter(row["DESCRIPTION"]))
        actual.append("spam" in row.get('SPAM', "") or "unusable" in row.get("STATUS", ""))
    if not (True in pred and False in pred):
        return None
    return confusionMatrix.create_confusionmatrix(actual, pred)

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
