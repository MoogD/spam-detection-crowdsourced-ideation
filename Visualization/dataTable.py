import matplotlib.pyplot as plt
import pandas as pd

import variables
from Helper import importDataHelper

def evaluationData_table(dataset):
    filterlistall = list(importDataHelper.readcsvdata(variables.evaluationpresultpath))
    data = []
    columns = ("Filter", "TP", "FP")
    num = 0
    pnum = 0
    nnum = 0
    gotdata = False

    fig, ax = plt.subplots(figsize=(10, 10))

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    for filter in filterlistall:
        if dataset in filter["Dataset"]:
            if not gotdata:
                num = filter["population"]
                pnum = filter["P"]
                nnum = filter["N"]
                gotdata = True
            if filter["Variable"] not in "None":
                if filter["TP"] in '' or filter["FP"] in '':
                    data.append([filter["Filter"] + ": " + filter["Variable"], 0, 0])
                else:
                    data.append([filter["Filter"] + ": " + filter["Variable"], int(filter["TP"]), int(filter["FP"])])
            else:
                if filter["TP"] in '' or filter["FP"] in '':
                    data.append([filter["Filter"], 0, 0])
                else:
                    data.append([filter["Filter"], int(filter["TP"]), int(filter["FP"])])

    df = pd.DataFrame(data, columns=columns)

    ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    fig.tight_layout()
    plt.title('Filter evaluation for ' + dataset + ' with ' + num + ' ideas (' + pnum + ' positives and ' + nnum + ' negatives)')
    plt.show()