from Helper import stringHelper
from Helper import importDataHelper
import time
import variables
######Todo Edit distance
###### ToDO levenshtein distance
###### Todo USE instead of LSA


# simple check for duplicate (find ideas with the same wordvector)
# returns (Bool, wordlist, ideamatrix)
# Bool = True for duplicates
# wordlist = list of strings containing all words that were used in any idea
# ideamatrix = list of lists. Each List represents one idea as a list of ints with [i] = x means wordlist[i] was x times in idea
def isduplicate(wordlist, idea):
    #resultsvec = stringHelper.getwordvec(wordlist, idea)
    # update matrix and return new matrix and wordlist, if there are new words in the idea
    ideawords = [word.lower() for word in stringHelper.getwordlist(idea)]
    cond = True
    duplicate = False
    for word in ideawords:
        wordlist[word] = wordlist.get(word, [])
        if wordlist[word] == []:
            cond = False
            wordlist[word] = [0]*len(list(wordlist.values())[0])

    if cond:
        i = 0
        while not duplicate and i < len(list(wordlist.values())[0]):
            for word in wordlist:
                if wordlist[word][i] is not ideawords.count(word):
                    duplicate = False
                    break
                else:
                    duplicate = True
            i += 1
    if not duplicate:
        for word in wordlist:
            wordlist[word].append(ideawords.count(word))

    return duplicate, wordlist




def filterduplikates(idealist, datapath=None, idealist2=None):
    wordlist = {}
    #ideamatrix = []
    duplicatelist = []
    newlist = []

    numberduplicates = 0
    nonduplicate = 0
    start = time.process_time_ns()
    if idealist2 is not None:
        for row in idealist2:
            # duplicateresults = isduplicate(wordlist, ideamatrix, row['DESCRIPTION'])
            duplicateresults = isduplicate(wordlist, row['DESCRIPTION'])
            wordlist = duplicateresults[1]
            # ideamatrix = duplicateresults[2]
    for row in idealist:
        row['TRIGGERED'] = []
        # duplicateresults = isduplicate(wordlist, ideamatrix, row['DESCRIPTION'])
        duplicateresults = isduplicate(wordlist, row['DESCRIPTION'])
        if duplicateresults[0]:
            numberduplicates += 1
            row['TRIGGERED'].append("duplicate")
            duplicatelist.append(row)
        else:
            nonduplicate += 1
            newlist.append(row)
        wordlist = duplicateresults[1]
        # ideamatrix = duplicateresults[2]
    duration = time.process_time_ns() - start
    # evaluate results:
    print("Duration duplicates: ", duration / 1000000000, "seconds")
    print("Duplicates: ", numberduplicates)
    print("Non-Duplicate Ideas: ", nonduplicate)
#    if len(duplicatelist) > 0 and datapath is not None:
#        importDataHelper.writecsvfile(datapath, duplicatelist[0].keys(), duplicatelist)
    return newlist