# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from utils.helpers import CfgError
import csv
from statistics import mean

def getJSONFromCSV(csvFilePath):
    data = []
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
        for rows in csvReader:
            data.append(rows)

    return data

def checkStability(valList: list, dev: float):
    meanVal = mean(valList)

    return all(
        [abs(x - meanVal) < dev * meanVal for x in valList]
    )

def checkPlausibility(leftInterval: list, rightInterval: list, dev: float):
    leftMean = mean(leftInterval)
    rightMean = mean(rightInterval)
    realGap = leftMean - rightMean
    return realGap > dev, realGap

def breakValidator(args):
    argDict = vars(args)
    if "-path_csv" not in argDict:
        raise CfgError("No 'path_csv' for break validator provided")
    csvPath = argDict["-path_csv"]
    if "-break_commit" not in argDict:
        raise CfgError("No 'break_commit' for break validator provided")
    if "-deviation" not in argDict:
        raise CfgError("No 'deviation' for break validator provided")

    breakCommit = argDict["-break_commit"]
    dev = float(argDict["-deviation"])

    commitMap = getJSONFromCSV(csvPath)

    breakId = int(
        [item['id'] for item in commitMap
            if item['hash'] == breakCommit][0]
        )

    leftInterval = [float(item['throughput']) for item in commitMap
            if int(item['id']) < breakId]
    rightInterval = [float(item['throughput']) for item in commitMap
            if int(item['id']) >= breakId]

    # first criterion: both intervals are stable
    isLeftStable = checkStability(leftInterval, dev)
    isRightStable = checkStability(rightInterval, dev)

    if not (isLeftStable and isRightStable):
        print("left interval is {}".format(
            "stable" if isLeftStable else "unstable"))
        print("right interval is {}".format(
            "stable" if isRightStable else "unstable"))
        return False
    
    # second criterion: left min > right max
    if not min(leftInterval) > max(rightInterval):
        print("pre-break interval does not majorize post-break")
        return False
    
    # third criterion: real gap between mean of intervals > expected deviation
    isPlausible, realGap = checkPlausibility(leftInterval, rightInterval, dev)
    if not isPlausible:
        print("realGap: {} more the expected deviation: {}".format(realGap, dev))
        return False
    
    return True
