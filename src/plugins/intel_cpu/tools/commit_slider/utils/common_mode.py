# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
import utils.helpers as util
from utils.subscription import SubscriptionManager
from utils.break_validator import validateBMOutput
from utils.break_validator import BmValidationError
import json
import os
from enum import Enum
import csv

class Mode(ABC):
    @staticmethod
    def factory(cfg):
        modeClassName = util.checkAndGetClassnameByConfig(
            cfg, "modeMap", "mode"
        )
        cl = util.checkAndGetSubclass(modeClassName, Mode)
        return cl(cfg)

    def __init__(self, cfg) -> None:
        self.checkCfg(cfg)
        self.commitPath = self.CommitPath()
        traversalClassName = util.checkAndGetClassnameByConfig(
            cfg, "traversalMap", "traversal"
        )
        traversalClass = util.checkAndGetSubclass(
            traversalClassName, self.Traversal
        )
        self.traversal = traversalClass(self)
        self.cfg = cfg
        logPath = util.getActualPath("logPath", cfg)
        self.commonLogger = util.setupLogger(
            "commonLogger", logPath, "common_log.log"
        )

    def isPerformanceBased(self):
        return False

    def createCash(self):
        # In common case we use json.
        # Create cash is overrided if we need special algo for caching.
        cp = util.getActualPath("cachePath", self.cfg)
        if not os.path.exists(cp):
            os.makedirs(cp)
        self.cachePath = os.path.join(cp, "check_output_cache.json")
        initCacheMap = {}
        try:
            with open(self.cachePath, "r+") as cacheDump:
                if self.cfg["clearCache"]:
                    cacheDump.truncate(0)
                    json.dump(initCacheMap, cacheDump)
                else:
                    try:
                        json.load(cacheDump)
                    except json.decoder.JSONDecodeError:
                        json.dump(initCacheMap, cacheDump)
        except FileNotFoundError:
            with open(self.cachePath, "w") as cacheDump:
                json.dump(initCacheMap, cacheDump)
        cacheDump.close()

    def getCommitIfCashed(self, commit):
        with open(self.cachePath, "r") as cacheDump:
            cacheData = json.load(cacheDump)
            cacheDump.close()
            if commit in cacheData:
                return True, cacheData[commit]
            else:
                return False, None

    def setCommitCash(self, commit, valueToCache, checkIfCashed=False):
        isCommitCashed, _ = self.getCommitIfCashed(commit)
        if isCommitCashed and checkIfCashed:
            raise util.CashError("Commit already cashed")
        else:
            with open(self.cachePath, "r+", encoding="utf-8") as cacheDump:
                cacheData = json.load(cacheDump)
                cacheData[commit] = valueToCache
                cacheDump.seek(0)
                json.dump(cacheData, cacheDump, indent=4)
                cacheDump.truncate()
                cacheDump.close()

    def getPseudoMetric(self, commit, cfg):
        raise NotImplementedError("getPseudoMetric() is not implemented")

    def compareCommits(self, c1: str, c2: str, cfg: map):
        raise NotImplementedError("compareCommits() is not implemented")

    def checkCfg(self, cfg):
        if not ("traversal" in cfg["runConfig"]):
            raise util.CfgError("traversal is not configured")

    def preliminaryCheck(self, list, cfg):
        # common checking if degradation exists
        if cfg["checkIfBordersDiffer"] and not self.checkIfListBordersDiffer(
                list, cfg):
            raise util.PreliminaryAnalysisError(
                "No degradation found: {i1} and {i2} don't differ".format(
                    i1=list[0], i2=list[-1]),
                util.PreliminaryAnalysisError.\
                    PreliminaryErrType.NO_DEGRADATION
                )

    def prepareRun(self, list, cfg):
        self.normalizeCfg(cfg)
        cfg["serviceConfig"] = {}
        # check prerun-cashed commits
        canReduce, newList = util.getReducedInterval(list, cfg)
        if canReduce:
            if (self.traversal.isComparative() and
                self.checkIfListBordersDiffer(newList, cfg) or
                not self.traversal.isComparative()):
                self.commonLogger.info(
                    "Initial interval reduced to cashed {c1}..{c2}".format(
                        c1=newList[0], c2=newList[-1])
                )
                list = newList
            elif self.traversal.isComparative():
                raise util.PreliminaryAnalysisError(
                    "No degradation for reduced interval: \
                    {i1} and {i2} don't differ".format(
                        i1=list[0], i2=list[-1]),
                    util.PreliminaryAnalysisError.\
                        PreliminaryErrType.NO_DEGRADATION
                    )
        else:
            self.preliminaryCheck(list, cfg)
        return list

    def normalizeCfg(self, cfg):
        # fetch paths for dlb job
        if cfg["dlbConfig"]["launchedAsJob"]:
            cfg["appPath"] = cfg["dlbConfig"]["appPath"]
        # switch off illegal check
        if not self.traversal.isComparative():
            cfg["checkIfBordersDiffer"] = False
        # apply necessary subscriptions for cfg
        subManager = SubscriptionManager(cfg)
        subManager.apply()
        if "modeName" in cfg["skipMode"]:
            errorHandlingMode = cfg["skipMode"]["modeName"]
            if errorHandlingMode == "skip":
                cfg["skipMode"]["flagSet"] = {}
                cfg["skipMode"]["flagSet"]["enableSkips"] = True
            elif (errorHandlingMode == "rebuild"):
                cfg["skipMode"]["flagSet"] = {}
                cfg["skipMode"]["flagSet"]["enableSkips"] = False
                cfg["skipMode"]["flagSet"]["enableRebuild"] = True
                cfg["skipMode"]["flagSet"]["switchOnSimpleBuild"] = True
                cfg["skipMode"]["flagSet"]["switchOnExtendedBuild"] = False
            else:
                raise util.CfgError(
                    "Error handling mode {} is not supported".format(errorHandlingMode)
                    )

    def postRun(self, list: list):
        util.returnToActualVersion(self.cfg)
        if "printCSV" in self.cfg\
                and self.cfg["printCSV"]\
                and self.isPerformanceBased():
            fields = ['linId', 'logId', 'hash', 'value'] 
            rows = []
            linearId = 0
            logId = 0
            for item in list:
                item = item.replace('"', "")
                isCommitCashed, value = self.getCommitIfCashed(item)
                if isCommitCashed:
                    row = [linearId, logId, item, value]
                    rows.append(row)
                    logId = logId + 1
                linearId = linearId + 1
            reportPath = util.getActualPath("logPath", self.cfg)
            reportPath = os.path.join(reportPath, "report.csv")
            with open(reportPath, 'w') as csvfile: 
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fields)
                csvwriter.writerows(rows)
        if self.isPerformanceBased():
            commitList = [{
                'id': int(list.index(item.cHash)),
                'hash': item.cHash,
                'throughput': self.getCommitIfCashed(item.cHash)[1]
            } for i, item in enumerate(self.commitPath.getList())]
            commitList = sorted(
                commitList,
                key=lambda el: el['id']
            )
            breakCommit = [item.cHash for item in self.commitPath.getList()\
                           if item.state == self.CommitPath.CommitState.BREAK][0]
            try:
                validateBMOutput(
                    commitList,
                    breakCommit,
                    self.cfg["runConfig"]["perfAppropriateDeviation"],
                    self.traversal.numericComparator()[0]
                )
            except BmValidationError as e:
                self.commitPath.metaInfo["postValidationPassed"] = False
                self.commitPath.metaInfo["reason"] = e.message

    def run(self, list, cfg) -> int:
        try:
            list = self.prepareRun(list, cfg)
            for i, item in enumerate(list):
                list[i] = item.replace('"', "")
            self.traversal.wrappedBypass(
                list, list, cfg
            )
            self.postRun(list)
        except util.PreliminaryAnalysisError as e:
            self.commitPath.metaInfo["preValidationPassed"] = False
            self.commitPath.metaInfo["reason"] = e.message

    def setOutputInfo(self, pathCommit):
        # override if you need more details in output representation
        pass

    def printResult(self):
        if not self.commitPath.metaInfo["preValidationPassed"]:
            print("Preliminary check failed, reason: {}".format(
                self.commitPath.metaInfo["reason"]
            ))
        elif not self.commitPath.metaInfo["postValidationPassed"]:
            print("Output results invalid, reason: {}".format(
                self.commitPath.metaInfo["reason"]
            ))
        else:
            for pathcommit in self.commitPath.getList():
                if pathcommit.state is not Mode.CommitPath.CommitState.DEFAULT:
                    print(self.getCommitInfo(pathcommit))

    def getCommitInfo(self, commit):
        # override if you need more details in output representation
        return "Break commit: {c}, state: {s}".format(
            c=commit.cHash, s=commit.state)

    def checkIfListBordersDiffer(self, list, cfg):
        return self.compareCommits(list[0], list[-1], cfg)

    class CommitPath:

        def __init__(self):
            self.commitList = []
            self.metaInfo = {
                    "preValidationPassed": True,
                    "postValidationPassed": True
                }

        def accept(self, traversal, commitToReport) -> None:
            traversal.visit(self, commitToReport)

        class CommitState(Enum):
            DEFAULT = 1
            BREAK = 2
            SKIPPED = 3
            IGNORED = 4

        class CommitSource(Enum):
            BUILDED = 1
            CASHED = 2

        class PathCommit:
            def __init__(self, cHash, state):
                self.cHash = cHash
                self.state = state

            def setupState(self, state):
                self.state = state

        def append(self, commit: PathCommit):
            if commit.cHash not in [x.cHash for x in self.commitList]:
                self.commitList.append(commit)

        def changeState(self, commit: str, state: CommitState):
            if commit in [x.cHash for x in self.commitList]:
                commitInd = [
                    i for i, elem in enumerate(self.commitList) if elem.cHash == commit
                    ][0]
                self.commitList[commitInd].setupState(state)
            else:
                raise Exception(
                    "Commit {} in not in commit path".format(commit))

        def pop(self):
            return self.commitList.pop(0)

        def getList(self):
            return self.commitList

    class Traversal(ABC):
        def bypass(self, curList, list, cfg) -> int:
            raise NotImplementedError()

        def numericComparator(self, leftVal: float=1, rightVal: float=-1, threshold: float=0) -> bool:
            # default numericComparator() represents traversal type for performance-based mode
            raise NotImplementedError()

        def wrappedBypass(self, curList, list, cfg) -> int:
            try:
                self.bypass(curList, list, cfg)
            except util.BuildError as be:
                if be.errType == util.BuildError.BuildErrType.TO_SKIP:
                    self.skipCommit(be.commit, curList, cfg)
                    self.wrappedBypass(curList, list, cfg)
                if be.errType == util.BuildError.BuildErrType.TO_IGNORE:
                    self.ignoreCommit(be.commit, curList, cfg)
                    self.wrappedBypass(curList, list, cfg)
                elif be.errType == util.BuildError.BuildErrType.TO_REBUILD:
                    cfg["extendBuildCommand"] = True
                    self.wrappedBypass(curList, list, cfg)
                else:
                    raise util.BuildError(
                        message = "error occured during handling",
                        errType = util.BuildError.BuildErrType.WRONG_STATE,
                        commit=be.commit
                        )


        def skipCommit(self, commit, curList, cfg):
            curList.remove(commit)
            pc = Mode.CommitPath.PathCommit(
                commit,
                Mode.CommitPath.CommitState.SKIPPED
            )
            self.mode.commonLogger.info(
                "Skipped commit {}".format(commit)
            )
            self.mode.setOutputInfo(pc)
            self.mode.commitPath.accept(self, pc)


        def ignoreCommit(self, commit, curList, cfg):
            curList.remove(commit)
            pc = Mode.CommitPath.PathCommit(
                commit,
                Mode.CommitPath.CommitState.IGNORED
            )
            self.mode.commonLogger.info(
                "Ignored commit {}".format(commit)
            )
            self.mode.setOutputInfo(pc)
            self.mode.commitPath.accept(self, pc)


        def visit(self, cPath, commitToReport):
            cPath.append(commitToReport)


        def prepBypass(self, curList, list, cfg):
            if (cfg["cachedPathConfig"]["enabled"] and
                cfg["cachedPathConfig"]["scheme"] == "optional"):
                # try to reduce interval by cashed borders
                canReduce, newList = util.getReducedInterval(curList, cfg)
                if canReduce:
                    if (self.isComparative() and
                        self.mode.checkIfListBordersDiffer(newList, cfg) or
                        not self.isComparative()):
                        self.mode.commonLogger.info(
                            "Interval {c1}..{c2} reduced to cashed {c1_}..{c2_}".format(
                                c1=curList[0], c2=curList[-1],
                                c1_=newList[0], c2_=newList[-1])
                        )
                        curList = newList
            i1 = list.index(curList[0])
            i2 = list.index(curList[-1])
            self.mode.commonLogger.info(
                "Check interval {i1}..{i2}".format(i1=i1, i2=i2)
            )
            self.mode.commonLogger.info(
                "Check commits {c1}..{c2}".format(c1=list[i1], c2=list[i2])
            )
            return curList

        def isComparative(self):
            # redefine for uncommon traversal
            return True


        def __init__(self, mode) -> None:
            self.mode = mode

    class FirstFailedVersion(Traversal):
        def __init__(self, mode) -> None:
            super().__init__(mode)

        def numericComparator(self, leftVal: float=1, rightVal: float=-1, threshold: float=0) -> bool:
            # default numericComparator() returns True, for current Traversal
            curRelation = rightVal / leftVal
            return (1 - curRelation) >= threshold, curRelation

        def bypass(self, curList, list, cfg) -> int:
            curList = self.prepBypass(curList, list, cfg)
            sampleCommit = curList[0]
            curLen = len(curList)
            if "sampleCommit" in cfg["serviceConfig"]:
                sampleCommit = cfg["serviceConfig"]["sampleCommit"]
            if curLen <= 2:
                isBad = self.mode.compareCommits(
                    sampleCommit, curList[0], cfg)
                breakCommit = curList[0] if isBad else curList[-1]
                self.mode.commitPath.changeState(
                    breakCommit,
                    Mode.CommitPath.CommitState.BREAK
                )
                return
            mid = (int)((curLen - 1) / 2)
            isBad = self.mode.compareCommits(
                    sampleCommit, curList[mid], cfg)
            if isBad:
                self.wrappedBypass(
                    curList[0 : mid + 1], list, cfg
                )
            else:
                self.wrappedBypass(
                    curList[mid :], list, cfg
                )

    class FirstFixedVersion(Traversal):
        def __init__(self, mode) -> None:
            super().__init__(mode)

        def numericComparator(self, leftVal: float=1, rightVal: float=-1, threshold: float=0) -> bool:
            # default numericComparator() returns False, for current Traversal
            curRelation = rightVal / leftVal
            return (curRelation - 1) >= threshold, curRelation

        def bypass(self, curList, list, cfg) -> int:
            curList = self.prepBypass(curList, list, cfg)
            sampleCommit = curList[0]
            curLen = len(curList)
            if "sampleCommit" in cfg["serviceConfig"]:
                sampleCommit = cfg["serviceConfig"]["sampleCommit"]
            if curLen <= 2:
                isBad = not self.mode.compareCommits(
                    sampleCommit, curList[0], cfg)
                breakCommit = curList[-1] if isBad else curList[0]
                self.mode.commitPath.changeState(
                    breakCommit,
                    Mode.CommitPath.CommitState.BREAK
                )
                return
            mid = (int)((curLen - 1) / 2)
            isBad = not self.mode.compareCommits(
                    sampleCommit, curList[mid], cfg)
            if isBad:
                self.wrappedBypass(
                    curList[mid :], list, cfg
                )
            else:
                self.wrappedBypass(
                    curList[0 : mid + 1], list, cfg
                )

    class AllBreakVersions(Traversal):
        def __init__(self, mode) -> None:
            super().__init__(mode)

        def bypass(self, curList, list, cfg) -> int:
            curList = self.prepBypass(curList, list, cfg)
            sampleCommit = curList[0]
            curLen = len(curList)
            if "sampleCommit" in cfg["serviceConfig"]:
                sampleCommit = cfg["serviceConfig"]["sampleCommit"]
            if curLen <= 2:
                isBad = self.mode.compareCommits(
                    sampleCommit, curList[0], cfg)
                breakCommit = curList[0] if isBad else curList[-1]
                self.mode.commitPath.changeState(
                    breakCommit,
                    Mode.CommitPath.CommitState.BREAK
                )
                lastCommit = list[-1]
                isTailDiffer = self.mode.compareCommits(
                    breakCommit, lastCommit, cfg)
                if isTailDiffer:
                    cfg["serviceConfig"]["sampleCommit"] = breakCommit
                    # to-do make copy without skip-commits
                    brIndex = list.index(breakCommit)
                    self.wrappedBypass(
                       list[brIndex :],
                       list, cfg
                    )
                return
            mid = (int)((curLen - 1) / 2)
            isBad = self.mode.compareCommits(
                    sampleCommit, curList[mid], cfg)
            if isBad:
                self.wrappedBypass(
                    curList[0 : mid + 1], list, cfg
                )
            else:
                self.wrappedBypass(
                    curList[mid :], list, cfg
                )

    class BruteForce(Traversal):
        def __init__(self, mode) -> None:
            super().__init__(mode)

        def bypass(self, curList, list, cfg) -> int:
            for commit in list:
                self.mode.commonLogger.info(
                    "Handle commit {}".format(commit))
                self.mode.getPseudoMetric(commit, cfg)

        def isComparative(self):
            return False
