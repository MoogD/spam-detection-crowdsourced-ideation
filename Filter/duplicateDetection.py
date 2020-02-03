from Helper import stringHelper
import xml.etree.ElementTree as ET
from Helper import importDataHelper
import time
import variables


# simple check for duplicate (find ideas with the same wordvector)
# returns (Bool, wordlist)
# Bool = True for duplicates
# wordlist = dictionary of string: with list of ints containing all words and the number of times it was used in the ideas were used in any idea
def isduplicate(wordlist, idea):
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




def filterduplikates(idealist, type, idealist2=None):
    wordlist = {}
    numberduplicates = 0
    nonduplicate = 0
    if idealist2 is not None:
        if type == "xml":
            ideas2 = idealist2.getroot()
        else:
            ideas2 = idealist
        for idea in ideas2:
            if type == "xml":
                for att in idea:
                    if att.tag == "{http://purl.org/gi2mo/ns#}content":
                        description = att.text
            else:
                description = idea["DESCRIPTION"]
            duplicateresults = isduplicate(wordlist, description)
            wordlist = duplicateresults[1]
    if type == "xml":
        ideas = idealist.getroot()
    else:
        ideas = idealist
    for idea in ideas:
        if type == "xml":
            for att in idea:
                if att.tag == "{http://purl.org/gi2mo/ns#}content":
                    description = att.text
        else:
            description = idea["DESCRIPTION"]
        duplicateresults = isduplicate(wordlist, description)
        if duplicateresults[0]:
            numberduplicates += 1
            if type == "xml":
                ET.SubElement(idea, "Duplicate", {"Duplicate": "yes"})
            else:
                idea["DUPLICATE"] = "yes"
        else:
            nonduplicate += 1
            if type == "xml":
                ET.SubElement(idea, "Duplicate", {"Duplicate": "no"})
            else:
                idea["DUPLICATE"] = "no"
        wordlist = duplicateresults[1]
    # evaluate results:
    print("Duplicates: ", numberduplicates)
    print("Non-Duplicate Ideas: ", nonduplicate)
    return idealist