import numpy as np
import matplotlib.pyplot as plt

import variables
from Helper import importDataHelper


def plot_Evaluation(dataset):
    N = 0
    filterlistall = list(importDataHelper.readcsvdata(variables.evaluationpresultpath))
    tplist = []
    fplist = []
    filterlistdataset = []

    fig, ax = plt.subplots(figsize=(10, 10))

    num = 0
    pnum = 0
    nnum = 0
    maxnum = 0
    temp = 0
    gotdata = False
    i = 0
    for filter in filterlistall:
        if dataset in filter["Dataset"]:
            i += 1
            if filter["Variable"] not in "None":
                filterlistdataset.append(str(i) + ": " + filter["Filter"] + ": " + filter["Variable"])
            else:
                filterlistdataset.append(str(i) + ": " + filter["Filter"])
            if not gotdata:
                num = filter["population"]
                pnum = filter["P"]
                nnum = filter["N"]
                gotdata = True
            N += 1
            if filter["TP"] in '':
                tplist.append(0)
            else:
                temp = int(filter["TP"])
                tplist.append(int(filter["TP"]))
            if filter["FP"] in '':
                fplist.append(0)
            else:
                temp += int(filter["FP"])
                fplist.append(int(filter["FP"]))
            if temp > maxnum:
                maxnum = temp
    ind = np.arange(N)
    p1 = plt.bar(ind, tplist)
    p2 = plt.bar(ind, fplist, bottom=tplist)
    plt.ylabel('Amount')
    plt.title(('Filter evaluation for '+ dataset + ' with '+ num + ' ideas (' + pnum + ' positives and ' + nnum + ' negatives)'))
    plt.xticks(ind, range(1, len(filterlistdataset)))
    plt.yticks(np.arange(0, maxnum, 20))
    plt.legend((p1[0], p2[0]), ('TP', 'FP'))
    print(filterlistdataset)
    plt.show()
