from Helper import stringHelper
# Filter for special data about the texts (lengt, wordcount...)


# return true if the idea has less then charcount letters (whitespaces removed)
def charcountlessfilter(idea, charcount):
    idea = idea.replace(' ', '')
    return len(idea) < charcount


# return true if the idea has more then charcount letters (whitespaces removed)
def charcountmorefilter(idea, charcount):
    idea = idea.replace(' ', '')
    return len(idea) > charcount


# return true if less then wordcount words in idea (words seperated by whitespaces
def wordcountfilter(idea, wordcount):
    wordlist = idea.split(' ')
    # remove empty strings and strings without letters:
    for word in wordlist:
        if word == '':
            wordlist.remove(word)
        elif stringHelper.isnotword(word):
            wordlist.remove(word)
    return len(wordlist) < wordcount

# check if idea is written in just UPPERCASE
def isuppercaseonly(idea):
    return idea.isupper()

textDataFilterList = [charcountlessfilter, charcountmorefilter, wordcountfilter, isuppercaseonly]