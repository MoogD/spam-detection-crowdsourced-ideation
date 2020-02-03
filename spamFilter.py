from Helper import importDataHelper
from Helper import stringHelper
from Filter import textDataFilter
from Filter import textContentFilter
from Filter import duplicateDetection
from Filter import bayes
import nltk
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import brown
import time
#TODO test different corpora
import variables

charcount = [5, 7, 10, 12, 15, 20, 25]
wordcount = [2, 3, 5, 7, 10, 15, 20]


def classifyidea(idea, unigram_tagger, st):
    idea, featuredata = applyfilter(idea, unigram_tagger, st)
    return idea

def classify_and_get_idea(idea, unigram_tagger, st):
    idea, filter, featuredata = applyfilter(idea, unigram_tagger, st)
    return idea, filter, featuredata

def applyfilter(idea, unigram_tagger, st):
    isspam = False
    featuredata = {}
    triggered = []
########################### apply textstructure filters: ###########################
    # check for number of characters in idea
    for count in charcount:
        if textDataFilter.charcountlessfilter(idea, count):
            triggered.append("Charcount")
            featuredata["less Chars " + str(count)] = 1
            isspam = True
        else:
            featuredata["less Chars " + str(count)] = 0

    # check number of words
    for count in wordcount:
        if textDataFilter.wordcountfilter(idea, count):
            triggered.append("Wordcount")
            isspam = True
            featuredata["less Words " + str(count)] = 1
        else:
            featuredata["less Words " + str(count)] = 0

    # filter ideas written in uppercase only
    if textDataFilter.isuppercaseonly(idea):
        triggered.append("Uppercase")
        isspam = True
        featuredata["uppercase only"] = 1
    else:
        featuredata["uppercase only"] = 0

########################### apply text content filters: ###########################
    # filter for special charsequences
    if textContentFilter.charseqfilter(idea):
        triggered.append("Charseq")
        isspam = True
        featuredata["charseq"] = 1
    else:
        featuredata["charseq"] = 0
    # filter for special sentence structures:
    if textContentFilter.sentencestructurefilter1(idea):
        triggered.append("sentencestructure1")
        isspam = True
        featuredata["sentencestructure1"] = 1
    else:
        featuredata["sentencestructure1"] = 0
    if textContentFilter.sentencestructurefilter2(idea):
        triggered.append("sentencestructure2")
        isspam = True
        featuredata["sentencestructure2"] = 1
    else:
        featuredata["sentencestructure2"] = 0
    if textContentFilter.sentencestructurefilter3(idea):
        triggered.append("sentencestructure3")
        isspam = True
        featuredata["sentencestructure3"] = 1
    else:
        featuredata["sentencestructure3"] = 0
    if textContentFilter.sentencestructurefilter4(idea):
        triggered.append("sentencestructure4")
        isspam = True
        featuredata["sentencestructure4"] = 1
    else:
        featuredata["sentencestructure4"] = 0
    if textContentFilter.isdefinition(idea):
        triggered.append("definition")
        isspam = True
        featuredata["definition"] = 1
    else:
        featuredata["definition"] = 0
    # filter for person names
    if textContentFilter.containsnames(idea, st):
        triggered.append("containsName")
        isspam = True
        featuredata["person name"] = 1
    else:
        featuredata["person name"] = 0

########################### apply NLP filter ###########################
########## Using Part-of-Speech tagger ##########
    # filter if idea contains atleast one noun
    if textContentFilter.withoutnoun(idea):
        triggered.append("without noun")
        isspam = True
        featuredata["no noun pos"] = 1
    else:
        featuredata["no noun pos"] = 0
    # filter if idea contains atleast one verb or adjective
    if textContentFilter.withoutverb(idea) and textContentFilter.withoutadjective(idea):
        triggered.append("without verb or adj")
        isspam = True
        featuredata["no verb or adj pos"] = 1
    else:
        featuredata["no verb or adj pos"] = 0
########## Using Statistical tagger trained on a corpus ##########
    # filter if idea contains atleast one noun
    if textContentFilter.withoutnoununigram(idea, unigram_tagger):
        triggered.append("without noun unigram")
        isspam = True
        featuredata["no noun unigram"] = 1
    else:
        featuredata["no noun unigram"] = 0
    # filter if idea contains atleast one verb or adjective
    if textContentFilter.withoutverbunigram(idea, unigram_tagger) and textContentFilter.withoutadjectiveunigram(idea, unigram_tagger):
        triggered.append("without verb or adj unigram")
        isspam = True
        featuredata["no verb or adj unigram"] = 1
    else:
        featuredata["no verb or adj unigram"] = 0
########################### add prediction ###########################
#    if isspam:
#        idea['TRIGGERED'] = idea['TRIGGERED'][:-2]
#        idea['PREDICTION'] = "Spam"
#    else:
#        idea['PREDICTION'] = "Ham"

    return idea, triggered, featuredata


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
