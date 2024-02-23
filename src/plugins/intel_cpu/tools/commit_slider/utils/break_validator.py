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
    realGap = abs(leftMean - rightMean) / leftMean
    return realGap > dev, realGap

def checkBreakLocality(leftInterval: list, rightInterval: list, dev: float):
    preBreakValue = leftInterval[-1]
    postBreakValue = rightInterval[0]
    realGap = abs(preBreakValue - postBreakValue) / preBreakValue
    return realGap > dev, realGap

def validateBMOutput(commitList: list, breakCommit: str, dev: float, isUpDownBreak: bool=True):
    breakId = int(
        [item['id'] for item in commitList
            if item['hash'] == breakCommit][0]
        )

    leftInterval = [float(item['throughput']) for item in commitList
            if int(item['id']) < breakId]
    rightInterval = [float(item['throughput']) for item in commitList
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
    majorizationCheck = min(leftInterval) > max(rightInterval) if isUpDownBreak\
        else max(leftInterval) < min(rightInterval)
    if not majorizationCheck:
        raise BmValidationError(
            "pre-break interval does not majorize post-break",
            BmValidationError.BmValErrType.MAJORIZATION_ERROR
        )
    
    # third criterion: real gap between mean of intervals > expected deviation
    isPlausible, realGap = checkPlausibility(leftInterval, rightInterval, dev)
    if not isPlausible:
        raise BmValidationError(
            "mean realGap: {} less than expected deviation: {}".format(
                realGap, dev
                ),
            BmValidationError.BmValErrType.LOW_GAP
        )

    # fourth criterion: gap between adjacent pre-break commit and break commit
    # must be more, than expected deviation
    isPlausible, realGap = checkBreakLocality(leftInterval, rightInterval, dev)
    if not isPlausible:
        raise BmValidationError(
            "local realGap: {} more than expected deviation: {}".format(
                realGap, dev
                ),
            BmValidationError.BmValErrType.LOW_LOCAL_GAP
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

    commitList = getJSONFromCSV(csvPath)

    try:
        validateBMOutput(commitList, breakCommit, dev)
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
        LOW_GAP = 2,
        # real gap is less, than expected
        LOW_LOCAL_GAP = 3
        # gap between pre-break and break is lower, than expected
    def __init__(self, message, errType):
        self.message = message
        self.errType = errType
    def __str__(self):
        return self.message