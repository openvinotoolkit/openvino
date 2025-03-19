# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
import json
from utils.cfg_manager import CfgManager

class TestData():
    __test__ = False
    actualDataReceived = False  # prevent slider running before config actualized

    def getTestCase():
        return TestData.TestCase.DeadTest

    def getTestName(self):
        raise NotImplementedError("getTestName() is not implemented")
    
    def fillActualData(self, markedVersionList):
        # redefine for non trivial cases
        self.start=markedVersionList[0]['commit']
        self.end=markedVersionList[-1]['commit']

    def formatConfig(self, content):
        # redefine for non trivial cases
        return CfgManager.multistepStrFormat(content, [
            {'p' : 'appCmd', 's' : "./{}".format(self.repoName)},
            {'p' : 'appPath', 's' : "tests/{}/build".format(self.repoName)},
            {'p' : 'buildPath', 's' : "tests/{}/build".format(self.repoName)},
            {'p' : 'gitPath', 's' : "tests/{}".format(self.repoName)},
            {'p' : 'start', 's' : self.start},
            {'p' : 'end', 's' : self.end}
        ])
    
    @staticmethod
    def checkTestSet():
        testCases = [td.getTestCase() for td in TestData.__subclasses__()]
        if len(testCases) != len(set(testCases)):
            raise TestError("Test cases don't differ correctly")
        elif TestData.TestCase.DeadTest in testCases:
            raise TestError("Test containing undefined getTestCase() found")
    
    @staticmethod
    def getTestByCase(tc):
        TestData.checkTestSet()
        foundTest = [td for td in TestData.__subclasses__()
                if td.getTestCase() == tc]
        if not foundTest:
            raise TestError("Test {} is not found".format(tc))
        return foundTest[0]

    @staticmethod
    def factory(testCase):
        return TestData.getTestByCase(testCase)(testCase)

    class TestCase(Enum):
        DeadTest = 0,
        FirstBadVersion = 1,
        FirstValidVersion = 2,
        BmValidatorStable = 3,
        BmValidatorSteppedBreak = 4,
        BmValidatorSteppedBreak2 = 5,
        BmBinarySearch = 6,
        BmBinarySearchUnstable = 7,
        BmNoDegradation = 8,
        BmUnstableDev = 9,
        BmWrongPath = 10,
        BmPathFound = 11,
        BmFirstFixed = 12,
        BmLatencyMetric = 13,
        ACModeData = 14,
        CustomizedLog = 15,
        MultiConfig = 16,
        ConfigMultiplicator = 17,
        MultiConfigWithKey = 18,
        AcModeDataBitwise = 19

    def requireTestData(self, reqLambda):
        # mapping json to test data holder
        with open("tests_res/tests_res.json") as cfgFile:
            rsc = json.load(cfgFile)
            reqLambda(self, rsc)

    def __init__(self):
        pass


class FirstBadVersionData(TestData):
    def getTestCase():
        return TestData.TestCase.FirstBadVersion

    def getTestName(self):
        return "FirstBadVersion"

    def __init__(self):
        from test_util import requireBinarySearchData
        self.requireTestData(
            requireBinarySearchData
        )

class CustomizedLogData(TestData):
    def getTestCase():
        return TestData.TestCase.CustomizedLog

    def getTestName(self):
        return "CustomizedLog"

    def __init__(self):
        from test_util import requireBinarySearchData
        self.requireTestData(
            requireBinarySearchData
        )

class MultiConfigData(TestData):
    def getTestCase():
        return TestData.TestCase.CustomizedLog

    def getTestName(self):
        return "MultiConfig"

    def __init__(self):
        from test_util import requireBinarySearchData
        self.requireTestData(
            requireBinarySearchData
        )

class ConfigMultiplicatorWithKeyData(TestData):
    def getTestCase():
        return TestData.TestCase.MultiConfigWithKey

    def getTestName(self):
        return "MultiConfigWithKey"

    def __init__(self):
        from test_util import requireBinarySearchData
        self.requireTestData(
            requireBinarySearchData
        )

class ConfigMultiplicatorData(TestData):
    def getTestCase():
        return TestData.TestCase.ConfigMultiplicator

    def getTestName(self):
        return "ConfigMultiplicatorByKey"

    def __init__(self):
        self.requireTestData(
            lambda td, rsc: [
                setattr(td, key, rsc[td.getTestName()][key])
                for key in [
            'multipliedCfg', 'testCfg'
            ]]
        )

class BenchmarkAppDataUnstable(TestData):
    def getTestCase():
        return TestData.TestCase.BmBinarySearchUnstable

    def getTestName(self):
        return "BmBinarySearchUnstable"

    def __init__(self):
        from test_util import requireBinarySearchData
        self.requireTestData(
            requireBinarySearchData
        )

class BenchmarkAppDataStable(TestData):
    def getTestCase():
        return TestData.TestCase.BmBinarySearchStable

    def getTestName(self):
        return "BmBinarySearchStable"

    def __init__(self):
        from test_util import requireBinarySearchData
        self.requireTestData(
            requireBinarySearchData
        )

class AcModeData(TestData):
    def getTestCase():
        return TestData.TestCase.ACModeData

    def getTestName(self):
        return "ACMode"

    def __init__(self):
        from test_util import requireBinarySearchData
        self.requireTestData(
            requireBinarySearchData
        )

class AcModeDataBitwise(TestData):
    def getTestCase():
        return TestData.TestCase.AcModeDataBitwise

    def getTestName(self):
        return "ACMode"

    def __init__(self):
        from test_util import requireBinarySearchData
        self.requireTestData(
            requireBinarySearchData
        )

class BenchmarkFirstFixedAppData(TestData):
    def getTestCase():
        return TestData.TestCase.BmFirstFixed

    def getTestName(self):
        return "BmFirstFixed"

    def __init__(self):
        from test_util import requireBinarySearchData
        self.requireTestData(
            requireBinarySearchData
        )

class BenchmarkMetricData(TestData):
    def getTestCase():
        return TestData.TestCase.BmLatencyMetric

    def getTestName(self):
        return "BmLatencyMetric"

    def formatConfig(self, content):
        # todo - use by-step replacement and super().formatConfig(content)
        return content.format(
            appCmd="./{}".format(self.repoName),
            appPath="tests/{}/build".format(self.repoName),
            buildPath="tests/{}/build".format(self.repoName),
            gitPath="tests/{}".format(self.repoName),
            start=self.start,
            end=self.end,
            metric=self.metric
        )

    def __init__(self, metric: str):
        from test_util import requireBinarySearchData
        self.requireTestData(
            requireBinarySearchData
        )
        self.metric = metric

class BenchmarkAppNoDegradationData(TestData):
    def getTestCase():
        return TestData.TestCase.BmNoDegradation

    def getTestName(self):
        return "BmNoDegradation"

    def __init__(self):
        from test_util import requireBinarySearchData
        self.requireTestData(
            requireBinarySearchData
        )

class BenchmarkAppUnstableDevData(TestData):
    def getTestCase():
        return TestData.TestCase.BmUnstableDev

    def getTestName(self):
        return "BmUnstableDev"

    def __init__(self):
        from test_util import requireBinarySearchData
        self.requireTestData(
            requireBinarySearchData
        )

class BenchmarkAppWrongPathData(TestData):
    def getTestCase():
        return TestData.TestCase.BmWrongPath

    def getTestName(self):
        return "BmWrongPath"

    def __init__(self):
        from test_util import requireBinarySearchData
        self.requireTestData(
            requireBinarySearchData
        )

class BenchmarkAppPathFoundData(TestData):
    def getTestCase():
        return TestData.TestCase.BmPathFound

    def getTestName(self):
        return "BmPathFound"

    def __init__(self):
        from test_util import requireBinarySearchData
        self.requireTestData(
            requireBinarySearchData
        )

class FirstValidVersionData(TestData):
    def getTestCase():
        return TestData.TestCase.FirstValidVersion

    def getTestName(self):
        return "FirstValidVersion"

    def __init__(self):
        from test_util import requireBinarySearchData
        self.requireTestData(
            requireBinarySearchData
        )

class BmStableData(TestData):
    def getTestCase():
        return TestData.TestCase.BmValidatorStable

    def getTestName(self):
        return "BmValidatorStable"

    def __init__(self):
        self.requireTestData(
            lambda td, rsc: [
                setattr(td, key, rsc[td.getTestName()][key])
                for key in [
            'bmOutputMap', 'breakCommit', 'dev'
            ]]
        )


class BmValidatorSteppedBreakData(TestData):
    # break commit exists, but provided deviation covers several stepped degradations,
    # as a result, binary search returns wrong result,
    # correct break can be found with lower deviation
    # │ xxxxx
    # │     xx
    # │      x
    # │      xx
    # │       x
    # │       xx
    # │        x
    # │        xxxxxxxxxxxx            <-- real degradation, low dev
    # │                    x
    # │                     xxxxxxxx   <-- false degradation, stepped effect
    # │
    # └──────────────────────────────
    def getTestCase():
        return TestData.TestCase.BmValidatorSteppedBreak

    def getTestName(self):
        return "BmValidatorSteppedBreak"

    def __init__(self):
        self.requireTestData(
            lambda td, rsc: [
                setattr(td, key, rsc[td.getTestName()][key])
                for key in [
            'bmOutputMap', 'wrongBreakCommit', 'realBreakCommit',
            'highDev', 'lowDev'
            ]]
        )

class BmValidatorSteppedBreakData2(TestData):
    # throughput degrades gradually,
    # results must be regarded as invalid
    # │   xxxxxxxxx
    # │           xxx              <-- first break
    # │             xxx
    # │               xxx          <-- second break
    # │                 xxx
    # │                   xxx      <-- third break
    # │                     xxx
    # │                       xxxxxxxxxxx
    # └────────────────────────────────────

    def getTestCase():
        return TestData.TestCase.BmValidatorSteppedBreak2

    def getTestName(self):
        return "BmValidatorSteppedBreak2"

    def __init__(self):
        self.requireTestData(
            lambda td, rsc: [
                setattr(td, key, rsc[td.getTestName()][key])
                for key in [
            'bmOutputMap', 'breakCommit', 'dev'
            ]]
        )


class TestError(Exception):
    __test__ = False
    pass