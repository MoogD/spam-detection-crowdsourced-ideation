from os import listdir
from os.path import isfile, join

from Helper import importDataHelper
import variables

def match_iui_challenges():
    unmatchedlist = list(importDataHelper.readcsvdata("Data/ImportsClassified/iui-export-ideas.csv"))
    print(len(unmatchedlist))
    challengelist = list(importDataHelper.readcsvdata("Data/ImportsClassified/ideas-with-challenges.csv"))
    print(len(challengelist))
    count_unmatched = 0
    count_matched = 0
    for idea in unmatchedlist:
        matched = False
        for idea2 in challengelist:
            if (idea["ID"] in idea2["ID"]):
                idea["CHALLENGE"] = idea2["CHALLENGE"]
                count_matched += 1
                matched = True
                break
        if not matched:
            count_unmatched += 1
    print(count_unmatched)
    print(count_matched)
    extend_challenge_db(unmatchedlist)

def add_all_ideas_toDB():
    for file in listdir(variables.importpathclassified):
        if isfile(join(variables.importpathclassified, file)):
            if ".csv" in file:
                extend_challenge_db(list(importDataHelper.readcsvdata(join(variables.importpathclassified, file))))
                print("finished: " + file)
            else:
                print("just csv supported right now")


def extend_challenge_db(idealist):
    challengelist = {}

    for file in listdir(variables.ideadbpath):
        if isfile(join(variables.ideadbpath, file)):
            filename = file.split(".")[0]
            challengelist[filename] = list(importDataHelper.readcsvdata(join(variables.ideadbpath, file)))

    for idea in idealist:
        idea["CHALLENGE"] = idea.get("CHALLENGE", "")
        if "cscw19-1" in idea["CHALLENGE"]:
            challengelist["TCO"] = challengelist.get("TCO", [])
            if not any(e['ID'] == idea['ID'] for e in challengelist["TCO"]):
                challengelist["TCO"].append(idea)
        elif "chi19s1" in idea["CHALLENGE"]:
            challengelist["TCO"] = challengelist.get("TCO", [])
            if not any(e['ID'] == idea['ID'] for e in challengelist["TCO"]):
                challengelist["TCO"].append(idea)
        elif "bionic" in idea["CHALLENGE"].lower():
            challengelist["bionicRadar"] = challengelist.get("bionicRadar", [])
            if not any(e['ID'] == idea['ID'] for e in challengelist["bionicRadar"]):
                challengelist["bionicRadar"].append(idea)
        elif "fabric" in idea["CHALLENGE"].lower():
            challengelist["fabricDisplay"] = challengelist.get("fabricDisplay", [])
            if not any(e['ID'] == idea['ID'] for e in challengelist["fabricDisplay"]):
                challengelist["fabricDisplay"].append(idea)
    for key in challengelist.keys():
        importDataHelper.writecsvfile(join(variables.ideadbpath, key + ".csv"), challengelist[key][0].keys(), challengelist[key])
        print("saved " + key)

def match_challenge(challenge):
    if "cscw19-1" in challenge.lower():
        return "TCO"
    elif "chi19s1" in challenge.lower():
        return "TCO"
    elif "bionic" in challenge.lower():
        return "bionicRadar"
    elif "fabric" in challenge.lower():
        return "fabricDisplay"
    else:
        return challenge

