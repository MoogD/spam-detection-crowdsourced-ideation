from Helper import importDataHelper, stringHelper
import variables

import nltk
from nltk.corpus import brown

def classify_noun_corpus():
    nouncorpus = read_noun_corpus()
    print("Enter \'y\' if noun is concrete, \'n\' if noun is abstract, \'skip\' to skip the word or \'stop\' to safe results and stop classifying")
    for noun in nouncorpus.keys():
        if "unclassified" in nouncorpus[noun]:
            answer = input(noun + ": ")
            if 'y' in answer:
                nouncorpus[noun] = "C"
            elif 'n' in answer:
                nouncorpus[noun] = "A"
            elif 'skip' in answer:
                pass
            else:
                importDataHelper.writecsvfiledict(variables.dbpath + 'NLPdata/NounDB.csv', nouncorpus.keys(),
                                                  nouncorpus)
                break


def extend_noun_corpus():
    idealist = list(importDataHelper.readcsvdata(variables.importpathclassified + 'cscw19-unapproved-ideas_import.csv'))
    nouncorpus = read_noun_corpus()
    unigram_tagger = prepare_tagger()
    for idea in idealist:
        nouns = get_Nouns(idea['DESCRIPTION'], unigram_tagger)
        for noun in nouns:
            if noun not in nouncorpus:
                nouncorpus[noun] = "unclassified"
    importDataHelper.writecsvfiledict(variables.dbpath + 'NLPdata/NounDB.csv', nouncorpus.keys(), nouncorpus)

def read_noun_corpus():
    nounlist = list(importDataHelper.readcsvdata(variables.dbpath + 'NLPdata/NounDB.csv'))
    # convert spam and ham word lists to dicts
    nouncorpus = {}
    for row in nounlist:
        nouncorpus.update(row)
    return nouncorpus


def get_Nouns(idea, unigram_tagger):
    nounlist = []
    taggedWords = nltk.pos_tag(stringHelper.getwordlist(idea))
    for word in taggedWords:
        if word[1] is not None and "NN" in word[1] and word[0].lower() not in nounlist:
            nounlist.append(word[0].lower())
    taggedWords = unigram_tagger.tag(stringHelper.getwordlist(idea))
    nonoun = True
    for word in taggedWords:
        if word[1] is not None and "NN" in word[1] and word[0].lower() not in nounlist:
            nounlist.append(word[0].lower())
    return  nounlist



def prepare_tagger():
    ### prepare unigram tagger with brown corpus
    brown_tagged_sents = brown.tagged_sents(
        categories=['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned',
                    'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction'])
    unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
    return unigram_tagger
