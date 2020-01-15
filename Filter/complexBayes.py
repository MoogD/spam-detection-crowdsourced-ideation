from Helper import importDataHelper
from Helper import stringHelper
import variables

import os

############ Sparse Binary Polynomial Hashing ############

# maximum number of words in phrase
n = 5

def gethamtokens(challenge=None, duplicates=False):
    # get list of ham words from old ideas
    if challenge is None:
        if duplicates:
            bayeshamwords = list(importDataHelper.readcsvdata(variables.complexbayesmixedpath + 'duplicateBayesHamToken.csv'))
        else:
            bayeshamwords = list(importDataHelper.readcsvdata(variables.complexbayesmixedpath + 'bayesHamToken.csv'))
    else:
        if duplicates:
            bayeshamwords = list(importDataHelper.readcsvdata(
                variables.complexbayeschallengebasedpath + challenge + '/duplicateBayesHamToken.csv'))
        else:
            bayeshamwords = list(importDataHelper.readcsvdata(variables.complexbayeschallengebasedpath + challenge + '/bayesHamToken.csv'))
    # convert spam and ham word lists to dicts
    hamdict = {}
    for row in bayeshamwords:
        hamdict.update(row)
    print("Old Ham Ideas (complex): ", hamdict['<IdeaCount>'])
    return hamdict

# get dict of spamword:
def getspamtokens(challenge=None, duplicates=False):
    # get list of spam words from old ideas
    if challenge is None:
        if duplicates:
            bayesspamwords = list(importDataHelper.readcsvdata(variables.complexbayesmixedpath + 'duplicateBayesSpamToken.csv'))
        else:
            bayesspamwords = list(importDataHelper.readcsvdata(variables.complexbayesmixedpath + 'bayesSpamToken.csv'))
    else:
        if duplicates:
            bayesspamwords = list(importDataHelper.readcsvdata(
                variables.complexbayeschallengebasedpath + challenge + '/duplicateBayesSpamToken.csv'))
        else:
            bayesspamwords = list(importDataHelper.readcsvdata(variables.complexbayeschallengebasedpath + challenge + '/bayesSpamToken.csv'))
    spamdict = {}
    for row in bayesspamwords:
        spamdict.update(row)
    print("Old Spam Ideas (complex): ", spamdict['<IdeaCount>'])
    return spamdict

def gettokenprobs(challenge=None, duplicates=False):
    if challenge is None:
        if duplicates:
            bayesphraseprobs = list(
                importDataHelper.readcsvdata(variables.complexbayesmixedpath + 'duplicateBayesTokenProbs.csv'))
        else:
            bayesphraseprobs = list(importDataHelper.readcsvdata(variables.complexbayesmixedpath + 'bayesTokenProbs.csv'))
    else:
        if duplicates:
            bayesphraseprobs = list(importDataHelper.readcsvdata(
                variables.complexbayeschallengebasedpath + challenge + '/duplicateBayesTokenProbs.csv'))
        else:
            bayesphraseprobs = list(importDataHelper.readcsvdata(variables.complexbayeschallengebasedpath + challenge + '/bayesTokenProbs.csv'))
    probdict = {}
    for row in bayesphraseprobs:
        probdict.update(row)
    return probdict

# update the wordcounts in hamword-/spamworddb with words from the new idea
def updatedb(idea, tokenlist):
    phraselist = stringHelper.getphraselist(idea)
    for phrase in phraselist:
        tokenlist[phrase] = int(tokenlist.get(phrase, 0)) + 1
    return tokenlist

def calculateprobs(spamwords, hamwords, nspam, nham):
#TODO Check if  <IdeaCount> should be excluded from calculated probs
    wordprobs = {}
    # maybe add the same for missed hamwords too (needed?)
    for word in spamwords.keys():
        spam = int(spamwords.get(word, 0))
        ham = 2*(int(hamwords.get(word, 0)))
        if (spam + ham) > 5:
            prob = max(0.000000000001, min(0.999999999999, (min(1, (spam/nspam)))/(min(1, ham/nham) + min(1, spam/nspam))))
            wordprobs[word] = prob
    return wordprobs

def gettokens(idea, wordprobs):
    phraselist = stringHelper.getphraselist(idea)
    worddict = {}
    worddictneutral = {}
    for word in phraselist:
#        if (len(worddict)< 15):
        prob = float(wordprobs.get(word.lower(), 0.4))
##### Try ignoring all neutral words #####
        if not (0.4 <= prob <= 0.6 ):
            worddict[word.lower()] = prob
        else:
            worddictneutral[word.lower()] = prob
#        else:
#            wordprob = float(wordprobs.get(word.lower(), 0.4))
#            min = 1.0
#            minword = ""
#            for item in worddict:
#                if (min > abs(worddict[item] - 0.5)):
#                    min = abs(worddict[item] - 0.5)
#                    minword = item
#            if (min < abs(wordprob - 0.5)):
#                worddict.pop(minword, 0)
#                worddict[word.lower()] = wordprob
##### check if non neutral words are available  #####
    if len(worddict) > 0:
        return worddict
    else:
        return worddictneutral

def combinedprob(tokenlist):
    prod = 1.0
    invprob = 1.0
    for token in tokenlist:
        prod = prod * float(tokenlist.get(token))
        invprob = invprob * (1-tokenlist.get(token))
    return prod / (prod + invprob)

def classify(idea, wordprobs):
    tokenlist = gettokens(idea, wordprobs)
    return combinedprob(tokenlist)


def trainbayes(idealist, challenge=None, delete=False, duplicates=False):
    if delete:
        bayesspamwords = {}
        bayeshamwords = {}
    else:
        bayesspamwords = getspamtokens(challenge, duplicates)
        bayeshamwords = gethamtokens(challenge, duplicates)
    nspam = int(bayesspamwords.pop("<IdeaCount>", 0))
    nham  = int(bayeshamwords.pop("<IdeaCount>", 0))
    for idea in idealist:
        if idea.get("STATUS", "") == "unusable":
            bayesspamwords = updatedb(idea['DESCRIPTION'], bayesspamwords)
            nspam += 1
        elif idea.get("STATUS", "") == "usable":
            nham += 1
            bayeshamwords = updatedb(idea['DESCRIPTION'], bayeshamwords)
        elif idea.get("STATUS", "") == "unreviewed" and "spam" in idea.get('SPAM', ""):
            bayesspamwords = updatedb(idea['DESCRIPTION'], bayesspamwords)
            nspam += 1
        elif idea.get("STATUS", "") == "unreviewed" and "ham" in idea.get('SPAM', ""):
            nham += 1
            bayeshamwords = updatedb(idea['DESCRIPTION'], bayeshamwords)
#        if "spam" in idea.get('SPAM', "") or "unusable" in idea.get("STATUS"):
#            bayesspamwords = updatedb(idea['DESCRIPTION'], bayesspamwords)
#            nspam += 1
#        else:
#            nham += 1
#            bayeshamwords = updatedb(idea['DESCRIPTION'], bayeshamwords)
    bayesspamwords["<IdeaCount>"] = nspam
    bayeshamwords["<IdeaCount>"] = nham
    if challenge is None:
        if duplicates:
            importDataHelper.writecsvfiledict(variables.complexbayesmixedpath + 'duplicateBayesSpamToken.csv',
                                              bayesspamwords.keys(), bayesspamwords)
            importDataHelper.writecsvfiledict(variables.complexbayesmixedpath + 'duplicateBayesHamToken.csv',
                                              bayeshamwords.keys(), bayeshamwords)
            probslist = calculateprobs(bayesspamwords, bayeshamwords, nspam, nham)
            importDataHelper.writecsvfiledict(variables.complexbayesmixedpath + 'duplicateBayesTokenProbs.csv', probslist.keys(),
                                              probslist)
        else:
            importDataHelper.writecsvfiledict(variables.complexbayesmixedpath + 'bayesSpamToken.csv', bayesspamwords.keys(), bayesspamwords)
            importDataHelper.writecsvfiledict(variables.complexbayesmixedpath + 'bayesHamToken.csv', bayeshamwords.keys(), bayeshamwords)
            probslist = calculateprobs(bayesspamwords, bayeshamwords, nspam, nham)
            importDataHelper.writecsvfiledict(variables.complexbayesmixedpath + 'bayesTokenProbs.csv', probslist.keys(), probslist)
    else:
        if not os.path.exists(variables.complexbayeschallengebasedpath + challenge):
            try:
                os.mkdir(variables.complexbayeschallengebasedpath + challenge)
            except OSError:
                print("Path for Challenge does not exist and could not be created")
        if duplicates:
            importDataHelper.writecsvfiledict(
                variables.complexbayeschallengebasedpath + challenge + '/duplicateBayesSpamToken.csv', bayesspamwords.keys(),
                bayesspamwords)
            importDataHelper.writecsvfiledict(
                variables.complexbayeschallengebasedpath + challenge + '/duplicateBayesHamToken.csv', bayeshamwords.keys(),
                bayeshamwords)
            probslist = calculateprobs(bayesspamwords, bayeshamwords, nspam, nham)
            importDataHelper.writecsvfiledict(
                variables.complexbayeschallengebasedpath + challenge + '/duplicateBayesTokenProbs.csv', probslist.keys(),
                probslist)
        else:
            importDataHelper.writecsvfiledict(variables.complexbayeschallengebasedpath + challenge + '/bayesSpamToken.csv', bayesspamwords.keys(),
                                              bayesspamwords)
            importDataHelper.writecsvfiledict(variables.complexbayeschallengebasedpath + challenge + '/bayesHamToken.csv', bayeshamwords.keys(),
                                              bayeshamwords)
            probslist = calculateprobs(bayesspamwords, bayeshamwords, nspam, nham)
            importDataHelper.writecsvfiledict(variables.complexbayeschallengebasedpath + challenge + '/bayesTokenProbs.csv', probslist.keys(),
                                              probslist)
    print(probslist)
