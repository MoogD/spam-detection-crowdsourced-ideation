import os

from Helper import importDataHelper
from Helper import stringHelper
import variables

######################################### Data for Bayes ###############################################################
# get dict of hamword:
#TODO check if <IdeaCount> should be excluded
def gethamtokens(challenge=None, duplicates=False):
    # get list of ham words from old ideas
    if challenge is None:
        if duplicates:
            bayeshamwords = list(importDataHelper.readcsvdata(variables.simplebayesmixedpath + 'duplicateBayesHamToken.csv'))
        else:
            bayeshamwords = list(importDataHelper.readcsvdata(variables.simplebayesmixedpath + 'bayesHamToken.csv'))
    else:
        if duplicates:
            bayeshamwords = list(
                importDataHelper.readcsvdata(variables.simplebayeschallengebasedpath + challenge + '/duplicateBayesHamToken.csv'))
        else:
            bayeshamwords = list(importDataHelper.readcsvdata(variables.simplebayeschallengebasedpath + challenge + '/bayesHamToken.csv'))
    # convert spam and ham word lists to dicts
    hamdict = {}
    for row in bayeshamwords:
        hamdict.update(row)
    print("Old Ham Ideas: ", hamdict['<IdeaCount>'])
    return hamdict

# get dict of spamword:
def getspamtokens(challenge=None, duplicates=False):
    # get list of spam words from old ideas
    if challenge is None:
        if duplicates:
            bayesspamwords = list(importDataHelper.readcsvdata(variables.simplebayesmixedpath + 'duplicateBayesSpamToken.csv'))
        else:
            bayesspamwords = list(importDataHelper.readcsvdata(variables.simplebayesmixedpath + 'bayesSpamToken.csv'))
    else:
        if duplicates:
            bayesspamwords = list(importDataHelper.readcsvdata(
                variables.simplebayeschallengebasedpath + challenge + '/duplicateBayesSpamToken.csv'))
        else:
            bayesspamwords = list(importDataHelper.readcsvdata(variables.simplebayeschallengebasedpath + challenge + '/bayesSpamToken.csv'))
    spamdict = {}
    for row in bayesspamwords:
        spamdict.update(row)
    print("Old Spam Ideas: ", spamdict['<IdeaCount>'])
    return spamdict

# get dict of spamprobabilities for each word:
def gettokenprobs(challenge=None, duplicates=False):
    if challenge is None:
        if duplicates:
            bayeswordprobs = list(importDataHelper.readcsvdata(variables.simplebayesmixedpath + 'duplicateBayesTokenProbs.csv'))
        else:
            bayeswordprobs = list(importDataHelper.readcsvdata(variables.simplebayesmixedpath + 'bayesTokenProbs.csv'))
    else:
        if duplicates:
            bayeswordprobs = list(importDataHelper.readcsvdata(
                variables.simplebayeschallengebasedpath + challenge + '/duplicateBayesTokenProbs.csv'))
        else:
            bayeswordprobs = list(importDataHelper.readcsvdata(variables.simplebayeschallengebasedpath + challenge + '/bayesTokenProbs.csv'))
    probdict = {}
    for row in bayeswordprobs:
        probdict.update(row)
    return probdict


# update the wordcounts in hamword-/spamworddb with words from the new idea
def updatedb(idea, wordlist):
    wordvec = stringHelper.getwordvec([], idea)
    for word in wordvec[0]:
        wordlist[word.lower()] = int(wordlist.get(word.lower(), 0)) + 1
    return wordlist

################################### classification with Bayes ##########################################################
# calculate the probability, that an is containing a word is spam
def calculateprobs(spamwords, hamwords, nspam, nham):
    wordprobs = {}
    # maybe add the same for missed hamwords too (needed?)
    for word in spamwords.keys():
        spam = int(spamwords.get(word, 0))
        ham = 2*(int(hamwords.pop(word, 0)))
        if (spam + ham) > 5:
            prob = max(0.000000000001, min(0.999999999999, (min(1, (spam/nspam)))/(min(1, ham/nham) + min(1, spam/nspam))))
            wordprobs[word] = prob
    return wordprobs

def gettokens(idea, wordprobs):
    wordvec = stringHelper.getwordvec([], idea)
    worddict = {}
    for word in wordvec[0]:
        if (len(worddict)< 15):
            worddict[word.lower()] = float(wordprobs.get(word.lower(), 0.4))
        else:
            wordprob = float(wordprobs.get(word.lower(), 0.4))
            min = 1.0
            minword = ""
            for item in worddict:
                if (min > abs(worddict[item] - 0.5)):
                    min = abs(worddict[item] - 0.5)
                    minword = item
            if (min < abs(wordprob - 0.5)):
                worddict.pop(minword, 0)
                worddict[word.lower()] = wordprob

    return worddict

# calculate and return the combined probability of a dict from tokens with probabilities:
def combinedprob(tokenlist):
    prod = 1.0
    invprob = 1.0
    for token in tokenlist:
        prod = prod * float(tokenlist.get(token))
        invprob = invprob * (1-tokenlist.get(token))
    return prod / (prod + invprob)

# calculate probs that an idea is a spam idea
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
#        if idea.get('SPAM') == "spam" or idea.get("STATUS") == "unusable":
#            bayesspamwords = updatedb(idea['DESCRIPTION'], bayesspamwords)
#            nspam += 1
#        else:
#            nham += 1
#            bayeshamwords = updatedb(idea['DESCRIPTION'], bayeshamwords)
    bayesspamwords["<IdeaCount>"] = nspam
    bayeshamwords["<IdeaCount>"] = nham
    if challenge is None:
        if duplicates:
            importDataHelper.writecsvfiledict(variables.simplebayesmixedpath + 'duplicateBayesSpamToken.csv',
                                              bayesspamwords.keys(), bayesspamwords)
            importDataHelper.writecsvfiledict(variables.simplebayesmixedpath + 'duplicateBayesHamToken.csv',
                                              bayeshamwords.keys(), bayeshamwords)
            probslist = calculateprobs(bayesspamwords, bayeshamwords, nspam, nham)
            importDataHelper.writecsvfiledict(variables.simplebayesmixedpath + 'duplicateBayesTokenProbs.csv', probslist.keys(),
                                              probslist)
        else:
            importDataHelper.writecsvfiledict(variables.simplebayesmixedpath + 'bayesSpamToken.csv', bayesspamwords.keys(), bayesspamwords)
            importDataHelper.writecsvfiledict(variables.simplebayesmixedpath + 'bayesHamToken.csv', bayeshamwords.keys(), bayeshamwords)
            probslist = calculateprobs(bayesspamwords, bayeshamwords, nspam, nham)
            importDataHelper.writecsvfiledict(variables.simplebayesmixedpath + 'bayesTokenProbs.csv', probslist.keys(), probslist)
    else:
        if not os.path.exists(variables.simplebayeschallengebasedpath + challenge):
            try:
                os.mkdir(variables.simplebayeschallengebasedpath + challenge)
            except OSError:
                print("Path for Challenge does not exist and could not be created")
        if duplicates:
            importDataHelper.writecsvfiledict(
                variables.simplebayeschallengebasedpath + challenge + '/duplicateBayesSpamToken.csv', bayesspamwords.keys(),
                bayesspamwords)
            importDataHelper.writecsvfiledict(
                variables.simplebayeschallengebasedpath + challenge + '/duplicateBayesHamToken.csv', bayeshamwords.keys(),
                bayeshamwords)
            probslist = calculateprobs(bayesspamwords, bayeshamwords, nspam, nham)
            importDataHelper.writecsvfiledict(
                variables.simplebayeschallengebasedpath + challenge + '/duplicateBayesTokenProbs.csv', probslist.keys(),
                probslist)
        else:
            importDataHelper.writecsvfiledict(variables.simplebayeschallengebasedpath + challenge + '/bayesSpamToken.csv', bayesspamwords.keys(),
                                          bayesspamwords)
            importDataHelper.writecsvfiledict(variables.simplebayeschallengebasedpath + challenge + '/bayesHamToken.csv', bayeshamwords.keys(),
                                          bayeshamwords)
            probslist = calculateprobs(bayesspamwords, bayeshamwords, nspam, nham)
            importDataHelper.writecsvfiledict(variables.simplebayeschallengebasedpath + challenge + '/bayesTokenProbs.csv', probslist.keys(),
                                          probslist)
    print(probslist)
