# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
import json

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
        return content.format(
            appCmd="./{}".format(self.repoName),
            appPath="tests/{}/build".format(self.repoName),
            buildPath="tests/{}/build".format(self.repoName),
            gitPath="tests/{}".format(self.repoName),
            start=self.start,
            end=self.end
        )
    
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
        BmValidatorStable = 3

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
                setattr(td,
                        key,
                        rsc[td.getTestName()][key])
                for key in [
            'bmOutputMap', 'breakCommit', 'dev'
            ]]
        )


class TestError(Exception):
    __test__ = False
    pass