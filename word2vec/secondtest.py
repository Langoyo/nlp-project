import gensim.downloader
from gensim.models import KeyedVectors

import csv
import nltk
import numpy as np
import sklearn
import math

wv = KeyedVectors.load("gnews300.wordvectors", mmap='r')
safewords = wv.key_to_index.keys()
quotes = list()


with open('train.csv', newline = '', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
    for row in reader:
        if not row[0]=="id":
            temp_dict = dict()
            temp_dict["id"] = row[0]
            temp_dict["rawquote"] = row[1]
            line = nltk.word_tokenize(row[1])
            temp_dict["tq"] = [word.lower() for word in line
                if word not in [".", ",", ";"]]
            temp_dict["author"] = row[2]
            
            quotes.append(temp_dict)


# get average-vector representations, store in dict but also a list of tuples (avg, author)
avgvecs = dict()
avgvecs["MWS"] = list()
avgvecs["HPL"] = list()
avgvecs["EAP"] = list()
trainsize = 2000
for i in range(len(quotes)):            
    teststring = quotes[i]["tq"]
    wordvecs = list()
    for word in teststring:
        if word in safewords:
            wordvecs.append(np.array(wv[word]))
    wordvecs = np.array(wordvecs)
    avgvec = np.average(wordvecs, axis=0)
    quotes[i]["avg"] = avgvec
    if i < trainsize:
        avgvecs[quotes[i]["author"]].append(avgvec)

times = 1500
total = 0
for i in range(3000, 4500):
    testvec = quotes[i]["avg"]
    simscores = dict()
    simscores["MWS"] = 0
    simscores["HPL"] = 0
    simscores["EAP"] = 0

    names = ["MWS", "HPL", "EAP"]
    for key in names:
        for quotevec in avgvecs[key]:
            sim = np.dot(testvec, quotevec) / (np.linalg.norm(testvec) * np.linalg.norm(quotevec))
            simscores[key] = simscores[key] + sim

    max = 0
    maxAuthor = 0
    for key in names:
        avgsim = simscores[key]/len(avgvecs[key])
        if(avgsim > max):
            max = avgsim
            maxAuthor = key
        
    if quotes[i]["author"] == maxAuthor:
        print("success!")
        total = total + 1

print(total/times)