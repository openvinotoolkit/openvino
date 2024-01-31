# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from utils.helpers import CfgError
import csv
from statistics import mean
from enum import Enum

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

def validateBMOutput(commitMap: map, breakCommit: str, dev: float):
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
        raise BmValidationError(
            "left interval is {}, right interval is {}".format(
                "stable" if isLeftStable else "unstable",
                "stable" if isRightStable else "unstable"
            ),
            BmValidationError.BmValErrType.UNSTABLE_INTERVAL
        )
    
    # second criterion: left min > right max
    if not min(leftInterval) > max(rightInterval):
        raise BmValidationError(
            "pre-break interval does not majorize post-break",
            BmValidationError.BmValErrType.MAJORIZATION_ERROR
        )
    
    # third criterion: real gap between mean of intervals > expected deviation
    isPlausible, realGap = checkPlausibility(leftInterval, rightInterval, dev)
    if not isPlausible:
        raise BmValidationError(
            "realGap: {} more the expected deviation: {}".format(
                realGap, dev
                ),
            BmValidationError.BmValErrType.LOW_GAP
        )

    return True

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

    try:
        validateBMOutput(commitMap, breakCommit, dev)
        print("Results are valid")
    except BmValidationError as e:
        print("Invalid benchmark results: {}".format(e.message))


class BmValidationError(Exception):
    class BmValErrType(Enum):
        UNSTABLE_INTERVAL = 0
        # supposed stable interval deviation exceeded limit
        MAJORIZATION_ERROR = 1
        # 'low'-value interval intersects with 'high'-value
        # e.g. [1, 0, 1, 1, 1] doesn't majorize [0, 1, 0, 0, 0]
        LOW_GAP = 2
        # real gap is less, than expected
    def __init__(self, message, errType):
        self.message = message
        self.errType = errType
    def __str__(self):
        return self.message