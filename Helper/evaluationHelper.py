def convertResults(resultdict):
    merke = resultdict["actual"].replace("[", '').replace("]", '').split(", ")
    resultdict["actual"] = []
    for x in merke:
        if "True" in x:
            resultdict["actual"].append(True)
        else:
            resultdict["actual"].append(False)

    resultdict["bayes"] = resultdict["bayes"].replace("[", '').replace("]", '').split(", ")
    resultdict["bayes"] = [float(x) for x in resultdict["bayes"]]

    resultdict["complexbayes"] = resultdict["complexbayes"].replace("[", '').replace("]", '').split(", ")
    resultdict["complexbayes"] = [float(x) for x in resultdict["complexbayes"]]

    resultdict["USE"] = resultdict["USE"].replace("[", '').replace("]", '').split(", ")
    merke = [x for x in resultdict["USE"]]
    bla = 0
    resultdict["USE"] = []
    for x in merke:
        if "(" in x:
            bla = int(x.replace("(", ""))
        elif ")" in x:
            resultdict["USE"].append((bla, float(x.replace(")", ""))))

    resultdict["linCLassifier"] = resultdict["linCLassifier"].replace("[", '').replace("]", '').split(", ")
    merke = [x for x in resultdict["linCLassifier"]]
    resultdict["linCLassifier"] = []
    for x in merke:
        if "(" in x:
            bla = int(x.replace("(", ""))
        elif ")" in x:
            resultdict["linCLassifier"].append((bla, float(x.replace(")", ""))))

    merke = resultdict["linClassCo"].replace("[", '').replace("]", '').replace("\n", '').split(" ")
    resultdict["linClassCo"] = []
    for x in merke:
        if x not in '':
            resultdict["linClassCo"].append(x.strip())
    resultdict["linClassCo"] = [float(x) for x in resultdict["linClassCo"]]

    merke = resultdict["Filter"].replace("\'", '')
    merke = merke.replace("{", '').replace("}", '').replace("]", ": ").split(": ")
    resultdict["Filter"] = {}
    key = ""
    for x in merke:
        if '[' in x:
            resultdict["Filter"][key] = x.replace("[", '').replace("]", '').split(", ")
            resultdict["Filter"][key] = [int(y) for y in resultdict["Filter"][key]]
        else:
            key = x.replace(", ", '')
    return resultdict

def get_filter_results(resultdict):
    for key in list(resultdict.keys()):
        if "less Chars" in key and not "15" in key:
            length = len(resultdict.pop(key, 0))
        elif "less Words" in key and not "3" in key:
            resultdict.pop(key, None)
    resultlist = [0]*length
    for i in range(len(resultlist)):
        for key in resultdict.keys():
            if resultdict[key][i] == 1:
                resultlist[i] = 1
                break
    return resultlist

