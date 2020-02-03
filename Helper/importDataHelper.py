import csv
import xml.etree.ElementTree as ET

########### read Data ###########
# returns CSV Data as list of Dicts
def readcsvdata(datapath):
    csvfile = open(datapath, 'r', newline='', encoding='utf-8')

    return csv.DictReader(csvfile, delimiter=',')

def readxmldata(datapath):
    tree = ET.parse(datapath)
#    return converttreetodict(tree)
    return ET.parse(datapath)

def converttreetodict(tree):
    root = tree.getroot()
    idealist = []
    ns = {'gi2mo': 'http://purl.org/gi2mo/ns#', 'oig': 'http://purl.org/innovonto/types#'}
    for child in root:
        idealist.append({"ID": child.attrib['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about'], "DESCRIPTION": child.find('gi2mo:content', ns).text, "CHALLENGE": "TCO"})
    return idealist



########### write Data ###########
# writes data to a csvfile
def writecsvfile(datapath, fieldnames,  datalist):
    csvfile = open(datapath, 'w+', newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in datalist:
        writer.writerow(row)


def writecsvfiledict(datapath, fieldnames,  datalist):
    csvfile = open(datapath, 'w+', newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    writer.writerow(datalist)
