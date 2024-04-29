# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import zipfile
import os
import csv
import shutil
from utils.helpers import CfgError


def getMap(hashPatternList, intervalPatternList):
    intervalPos = 2  # as in 'Check interval i1..i2' pattern
    hashMap = {}

    for i, x in enumerate(intervalPatternList):
        leftHash = hashPatternList[i].split()\
            [intervalPos].split("..")[0]
        rightHash = hashPatternList[i].split()\
            [intervalPos].split("..")[1]

        leftInd = (x.split()[2]).split("..")[0]
        rightInd = (x.split()[2]).split("..")[1]

        hashMap[leftInd] = leftHash
        hashMap[rightInd] = rightHash

    return hashMap

def extractPatterns(dirName):
    data = ""

    with open(os.path.join(dirName, "logcommon_log.log")) as file:
        data = file.read()
    intervalPattern = "[A-Za-z0-9]*\.\.[A-Za-z0-9]*"

    pattern = "Check commits {}".format(intervalPattern)
    stats_re = re.compile(pattern, re.MULTILINE | re.DOTALL)
    hashPatternList = stats_re.findall(data)

    pattern = "Check interval {}".format(intervalPattern)
    stats_re = re.compile(pattern, re.MULTILINE | re.DOTALL)
    intervalPatternList = stats_re.findall(data)

    return hashPatternList, intervalPatternList

def prepareCSVData(hashMap, dirName):
    throughputPattern = "Throughput:\s*([0-9]*[.][0-9]*)\s*FPS"
    csvData = []

    for k, v in hashMap.items():
        logFileName = "logcommit_{}.log".format(v)

        if logFileName not in os.listdir(dirName):
            raise LookupError("No {} in logs".format(logFileName))

        with open(os.path.join(dirName, logFileName)) as logFile:
            data = logFile.read()

            foundThroughput = re.search(
                throughputPattern, data, flags=re.MULTILINE
                ).group(1)

            csvData.append({
                "id": k,
                "hash": v,
                "throughput": foundThroughput
            })

    csvData.sort(key=lambda x: int(x['id']))

    return csvData

def makeCSV(csvData):
    fields = ['id', 'hash', 'throughput']
    rows = []

    for item in csvData:
        row = [item['id'], item['hash'], item['throughput']]
        rows.append(row)

    with open("csv_report.csv", 'w') as csvfile: 
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

def logParser(args, zipName="archive", dirName = "artifacts"):
    args = vars(args)
    if "-path" not in args:
        raise CfgError("No 'path' for log parser provided")
    elif "-zip_name" in args:
        zipName = args["-zip_name"]
    path = str(args["-path"])

    clearArtifacts = False

    if path.endswith('.zip'):
        clearArtifacts = True
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(dirName)
            dirName = os.path.join(dirName, zipName)
    else:
        dirName = path
    
    hashPatternList, intervalPatternList = extractPatterns(dirName)
    hashMap = getMap(hashPatternList, intervalPatternList)
    csvData = prepareCSVData(hashMap, dirName)
    makeCSV(csvData)

    if clearArtifacts:
        shutil.rmtree(dirName)
