
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
punctuations = ['.', ',', '?', '!']

#Check if a word has at least one letter
def isnotword(word):
    isnotword = True
    for letter in word:
        if letter.lower() in letters:
            isnotword = False
            break
    return isnotword

### TODO lower words
def getwordlist(idea):
    wordlist = idea.replace('.', ' ').replace(',', ' ').replace('\"', ' ').replace('\'', ' ').replace('-', ' ').\
        replace('?', ' ').replace('!', ' ').replace(':', ' ').replace('”', ' ').replace('“', ' ').replace('—', ' ').\
        replace(';', ' ').split()
    for word in wordlist:
        if word == '':
            wordlist.remove(word)
    return wordlist

## get the first sentence from an Idea (without subsentences)
# does not work for some nested brackets
def getmainsentence(idea):
    subsentencecom = False
    subsentencebracket = 0
    words = idea.split()
    sentence = []
    for word in words:
        ## remove all words in subsentence
        if "," in word:
            newword = word.replace(',', ' ').split()
            if subsentencecom and len(newword) > 1:
                sentence.append(newword[1])
            elif not subsentencecom and len(newword) > 0:
                sentence.append(newword[0])
            subsentencecom = not subsentencecom
        elif "(" in word:
            subsentencebracket += 1
            if word[0] is not '(' and ')' not in word:
                sentence.append(word.replace('(', ' ').split()[0])
            elif ')' in word:
                subsentencebracket -= 1
                if word[len(word) - 1] is not ')':
                    sentence.append(word.replace(')', ' ').split()[1])
        elif '.' in word:
            if (not word[0] is '.') and not (subsentencecom or subsentencebracket > 0):
                sentence.append(word.replace('.', ' ').split()[0])
            return sentence
        elif not (subsentencecom or subsentencebracket > 0):
            sentence.append(word)
    return sentence

def getsentencelist(idea):
    sentencelist = []
    wordlist = idea.split()
    i = 0
    for word in wordlist:
        i += 1
        if '.' in word and '...' not in word and '..' not in word:
            sentencelist.append(wordlist[:i])
            wordlist = wordlist[i:]
            i = 0
    if len(wordlist) > 0:
        sentencelist.append(wordlist)
    return sentencelist


def getphraselist(idea):
    wordlist = getwordlist(idea)
    phraselist = []
    while len(wordlist) > 0:
        if len(wordlist) >= 5:
            phraselist += createphrases([wordlist[0].lower(), wordlist[1].lower(), wordlist[2].lower(), wordlist[3].lower(), wordlist[4].lower()])
        elif len(wordlist) == 4:
            phraselist += createphrases([wordlist[0].lower(), wordlist[1].lower(), wordlist[2].lower(), wordlist[3].lower()])
        elif len(wordlist) == 3:
            phraselist += createphrases([wordlist[0].lower(), wordlist[1].lower(), wordlist[2].lower()])
        elif len(wordlist) == 2:
            phraselist += createphrases([wordlist[0].lower(), wordlist[1].lower()])
        else:
            phraselist.append((wordlist[0].lower() + ", , , , ,"))
        wordlist.pop(0)

    return phraselist

def createphrases(wordlist):
    phraselist = []
    count = 2**(len(wordlist))
    test = 1
    for word in wordlist:
        for i in range(0, (2**(len(wordlist)-1))):
            if test == 1:
                phraselist.append(word + ", ")
            elif i%count < count//2:
               phraselist[i] += word + ", "
            else:
                phraselist[i] += ", "
        count = count//2
        test = 2

    x = 5 - len(phraselist[0])
    merkelist = []
    for i in range(0,x+1):
        merkelist.append("<!PlaceHolder!>")
    if (x > 0):
        for i in range(0, 2**(len(wordlist)-1)):
            phraselist[i] += merkelist

    return phraselist

def getwordvec(words, idea):
    wordlist = words.copy()
    wordlistidea = getwordlist(idea)
    wordvec = [0]*len(wordlist)
    for word in wordlistidea:
        try:
            wordvec[wordlist.index(word.lower())] += 1
        except (ValueError):
            wordlist.append(word.lower())
            wordvec.append(1)
    return (wordlist, wordvec)

# remove all single puncutation characters from string
def removepunctuation(idea):
    i = 0
    for char in idea:
        if (char in punctuations):
            if (idea[i-1] in letters or idea[i-1] in numbers or idea[i-1] == ' '):
                if (len(idea) <= i + 1):
                    idea = idea[:i]
                    i -= 1
                elif idea[i + 1] == ' ' or idea[i + 1] in letters or idea[i + 1] in numbers:
                    idea = idea[:i] + idea[i+1:]
                    i -= 1
        i += 1
    return idea