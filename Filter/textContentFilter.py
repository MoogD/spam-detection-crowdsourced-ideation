from Helper import stringHelper
import nltk


# spamindicator = ['...', '..']
enumeration_indicator = ['i.e', 'e.g', '.or', ' or.', ' or ', ' and.', ' and ', ' etc.', 'etc ', ',']
# TODO test amount of commata
# TODO check directory for isenumeration function
def isenumeration(wordlist):
    for word in wordlist:
        for enum in enumeration_indicator:
            if enum in word:
                return True
    return False

# TODO exlude lists/enumerations (Done? Testing)
# TODO exclude '...' when it is the last word (Done? Testing)
# TODO check if ideas with more then one  ... can be labeled without enumeration test
# check for enumeration: count amount of ... in idea and check if every sentence with ... has a enumeration indicator
def charseqfilter(idea):
    if '...' in idea:
        indicator_count = idea.count('...')
        if indicator_count > 1:
            return True
        sentences = stringHelper.getsentencelist(idea)
        # return false if ... is in the last word of the idea (no indicater for google search)
        if '...' in sentences[len(sentences)-1][len(sentences[len(sentences)-1])-1]:
            return False
        # return false if ... is used in an enumeration
        for sentence in sentences:
            i = 0
            for word in sentence:
                if "..." in word:
                    if isenumeration(sentence):
                        return False
#                        indicator_count -= 1
                i += 1
#        if indicator_count > 0:
        return True
    return False

########## Check for definitions: 'A ... is', 'The ... is' ##########
### Check if all covered with isdefinition()
def sentencestructurefilter1(idea):
    wordlist = stringHelper.getmainsentence(idea)
    return len(wordlist) > 2 and (wordlist[0].lower() == "a" and wordlist[2].lower() == "is")


def sentencestructurefilter2(idea):
    wordlist = stringHelper.getmainsentence(idea)
    return len(wordlist) > 3 and (wordlist[0].lower() == "a" and wordlist[3].lower() == "is")


def sentencestructurefilter3(idea):
    wordlist = stringHelper.getmainsentence(idea)
    return len(wordlist) > 2 and (wordlist[0].lower() == "the" and wordlist[2].lower() == "is")


def sentencestructurefilter4(idea):
    wordlist = stringHelper.getmainsentence(idea)
    return len(wordlist) > 3 and (wordlist[0].lower() == "the" and wordlist[3].lower() == "is")


### Might cause many fp !!!
def isdefinition(idea):
    wordlist = nltk.pos_tag(stringHelper.getmainsentence(idea))
    if len(wordlist) > 0 and "DT" in wordlist[0][1]:
        i = 1
        while i < len(wordlist) and "NN" in wordlist[i][1]:
            i += 1
        if len(wordlist) > i + 1 and i > 1 and "VB" in wordlist[i][1] and "DT" in wordlist[i+1][1]:
            return True
    i = 0
    while i < len(wordlist) and "NN" in wordlist[i][1]:
        i += 1
    if len(wordlist) > i + 1 and i > 0 and "VB" in wordlist[i][1] and "DT" in wordlist[i+1][1]:
        return True
    return False

### Stanford name tagger
def containsnames(idea, st):
    taggedwords = st.tag(idea.split())
    for word in taggedwords:
        if word[1] == "PERSON":
            return True
    return False

### pos tagger
### Check if unknown words should be treaded as noun or not
def withoutnoun(idea):
    taggedWords = nltk.pos_tag(stringHelper.getwordlist(idea))
    nonoun = True
    for word in taggedWords:
        if "NN" in word[1]:
            return False
        elif "None" in word[1]:
            nonoun = False
    return nonoun


def withoutverb(idea):
    taggedWords = nltk.pos_tag(stringHelper.getwordlist(idea))
    for word in taggedWords:
        if "VB" in word[1]:
            return False
    return True


def withoutadjective(idea):
    taggedWords = nltk.pos_tag(stringHelper.getwordlist(idea))
    for word in taggedWords:
        if "JJ" in word[1]:
            return False
    return True

## unigram tagger
### Check if unknown words should be treaded as noun or not
def withoutnoununigram(idea, tagger):
    taggedWords = tagger.tag(stringHelper.getwordlist(idea))
    nonoun = True
    for word in taggedWords:
        if word[1] is None:
            nonoun = False
        elif "NN" in word[1]:
            return False
    return nonoun

def withoutverbunigram(idea, tagger):
    taggedWords = tagger.tag(stringHelper.getwordlist(idea))
    for word in taggedWords:
        if word[1] is not None and ("VB" in word[1] or "MD" in word[1] or "BE" in word[1]):
            return False
    return True

def withoutadjectiveunigram(idea, tagger):
    taggedWords = tagger.tag(stringHelper.getwordlist(idea))
    for word in taggedWords:
        if word[1] is not None and "JJ" in word[1]:
            return False
    return True

textContentFilterlist = [charseqfilter, sentencestructurefilter1, sentencestructurefilter2, sentencestructurefilter3, sentencestructurefilter4, isdefinition, containsnames, withoutnoun, withoutverb, withoutadjective, withoutnoununigram, withoutverbunigram, withoutadjectiveunigram]