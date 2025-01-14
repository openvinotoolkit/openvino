# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from utils.helpers import fetchAppOutput, getActualPath
from utils.helpers import getMeaningfullCommitTail, extractModelPath
from utils.helpers import handleCommit, getBlobDiff
from utils.helpers import getCommitLogger, CashError, CfgError,\
CmdError, PreliminaryAnalysisError
from utils.break_validator import checkStability
import re
import shutil
from utils.common_mode import Mode


class NopMode(Mode):
    # helpful with mode-ignorant traversal (brute force)
    def __init__(self, cfg):
        self.msg = "default"
        super().__init__(cfg)

    def checkCfg(self, cfg):
        if "msg" in cfg["runConfig"]:
            self.msg = cfg["runConfig"]["msg"]
        super().checkCfg(cfg)

    def getPseudoMetric(self, commit, cfg):
        commit = commit.replace('"', "")
        commitLogger = getCommitLogger(cfg, commit)
        self.commonLogger.info("New commit: {commit}".format(
            commit=commit)
        )
        handleCommit(commit, cfg)
        checkOut = fetchAppOutput(cfg, commit)
        commitLogger.info(checkOut)
        return

    def printResult(self):
        print(self.msg)
        self.outLogger.info(self.msg)


class CheckOutputMode(Mode):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.createCash()

    def checkCfg(self, cfg):
        super().checkCfg(cfg)
        if not ("stopPattern" in cfg["runConfig"]):
            raise CfgError("stopPattern is not configured")

    def compareCommits(self, lCommit: str, rCommit: str, cfg: map):
        isLeftBorderFailed = bool(self.getPseudoMetric(lCommit, cfg))
        isRightBorderGood = not self.getPseudoMetric(rCommit, cfg)
        curCommit = rCommit.replace('"', "")
        commitLogger = getCommitLogger(cfg, curCommit)
        commitLogger.info(
            "Commit {c} is {status}".format(
                status=("good" if isRightBorderGood else "bad"),
                c=rCommit)
        )
        return isLeftBorderFailed == isRightBorderGood

    def getPseudoMetric(self, commit, cfg):
        commit = commit.replace('"', "")
        checkOut = ""
        commitLogger = getCommitLogger(cfg, commit)
        isCommitCashed, cashedOutput = self.getCommitIfCashed(commit)
        pc = Mode.CommitPath.PathCommit(
            commit,
            Mode.CommitPath.CommitState.DEFAULT
        )
        self.setOutputInfo(pc)
        self.commitPath.accept(self.traversal, pc)
        if isCommitCashed:
            logMsg = "Cashed commit - {commit}".format(commit=commit)
            self.commonLogger.info(logMsg)
            commitLogger.info(logMsg)
            checkOut = cashedOutput
        else:
            self.commonLogger.info("New commit: {commit}".format(
                commit=commit)
            )
            handleCommit(commit, cfg)
            checkOut = fetchAppOutput(cfg, commit)
            commitLogger.info(checkOut)
            self.setCommitCash(commit, checkOut)
        stopPattern = cfg["runConfig"]["stopPattern"]
        isFound = re.search(stopPattern, checkOut)
        if isFound is None:
            isFound = False
        return isFound


class BenchmarkAppPerformanceMode(Mode):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.perfRel = 0
        self.createCash()

    def isPerformanceBased(self):
        return True

    def prepareRun(self, list, cfg):
        super().prepareRun(list, cfg)
        sampleCommit = list[0]
        sampleCommit = sampleCommit.replace('"', "")
        self.commonLogger.info(
            "Prepare sample commit - {commit}".format(commit=sampleCommit)
        )
        commitLogger = getCommitLogger(cfg, sampleCommit)
        foundThroughput = 0
        isCommitCashed, cashedThroughput = self.getCommitIfCashed(sampleCommit)
        if isCommitCashed:
            logMsg = "Cashed commit - {commit}".format(commit=sampleCommit)
            self.commonLogger.info(logMsg)
            commitLogger.info(logMsg)
            foundThroughput = cashedThroughput
        else:
            handleCommit(sampleCommit, cfg)
            output = fetchAppOutput(cfg, sampleCommit)
            commitLogger.info(output)
            foundThroughput = re.search(
                self.outPattern, output, flags=re.MULTILINE
            ).group(1)
            self.setCommitCash(sampleCommit, float(foundThroughput))
        self.sampleThroughput = float(foundThroughput)
        return list

    def checkCfg(self, cfg):
        super().checkCfg(cfg)
        if not ("perfAppropriateDeviation" in cfg["runConfig"]):
            raise CfgError("Appropriate deviation is not configured")
        else:
            self.apprDev = cfg["runConfig"]["perfAppropriateDeviation"]
        if ("metric" in cfg["runConfig"]):
            self.outPattern = self.specifyMetric(cfg["runConfig"]["metric"])
        else:
            self.outPattern = self.specifyMetric()


    def specifyMetric(self, metric: str = "throughput"):
        if metric in [
            "throughput",
            "latency:max",
            "latency:min",
            "latency:median",
            "latency:average"]:
            spec = metric.split(":")
            idStr = "FPS"
            if len(spec) == 2:
                spec = spec[1]
                idStr = "ms"
            else:
                spec = spec[0]
            spec = spec.title()
            res = r'{spec}:\s*([0-9]*[.][0-9]*)\s*{idStr}'.format(
                spec=spec, idStr=idStr)
            return res
        raise CfgError("Benchmark metric {} is not supported".format(metric))

    def preliminaryCheck(self, list, cfg):
        # model path checking
        if cfg["preliminaryCheckCfg"]["checkBenchmarkModelPath"]:
            cmdStr = cfg["appCmd"]
            matcher = re.search(
                r"benchmark.*-m[\s*]([^\S]*)",
                cmdStr,
                flags=re.MULTILINE
                )
            if matcher is not None:
                # pass if app is not openvino benchmark_app
                try:
                    modelPath = extractModelPath(cmdStr)
                    if not os.path.isfile(modelPath):
                        raise PreliminaryAnalysisError(
                            "path {modelPath} does not exist, check config".format(
                                modelPath=modelPath
                            ),
                            PreliminaryAnalysisError.PreliminaryErrType.WRONG_COMMANDLINE
                        )
                except (IndexError, ValueError):
                    raise PreliminaryAnalysisError(
                        "commandline '{cmdStr}' is not correct, check config".format(
                            cmdStr=cmdStr
                        ),
                        PreliminaryAnalysisError.PreliminaryErrType.WRONG_COMMANDLINE
                    )

        # common if-degradation-exists check
        super().preliminaryCheck(list, cfg)

        # performance - specific check if results for borders are stable,
        isLeftStable = not cfg["preliminaryCheckCfg"]["leftCheck"] or\
            self.preliminaryStabilityCheck(list[0], cfg)
        isRightStable = not cfg["preliminaryCheckCfg"]["rightCheck"] or\
            self.preliminaryStabilityCheck(list[-1], cfg)
        if (not isLeftStable or not isRightStable):
            raise PreliminaryAnalysisError(
                "{lCommit} is {lStable}, {rCommit} is {rStable}".format(
                    lCommit=list[0],
                    rCommit=list[-1],
                    lStable="stable" if isLeftStable else "unstable",
                    rStable="stable" if isRightStable else "unstable"
                ),
                PreliminaryAnalysisError.PreliminaryErrType.UNSTABLE_APPLICATION
                )

    def compareCommits(self, lCommit: str, rCommit: str, cfg: map):
        leftThroughput = self.getPseudoMetric(lCommit, cfg)
        rightThroughput = self.getPseudoMetric(rCommit, cfg)
        isBad, curRel = self.traversal.numericComparator(
            leftThroughput, rightThroughput, self.apprDev
        )
        if isBad:
            self.perfRel = curRel
        curCommit = rCommit.replace('"', "")
        commitLogger = getCommitLogger(cfg, curCommit)
        commitLogger.info("Performance relation is {rel}".format(rel=curRel))
        commitLogger.info(
            "Commit is {status}".format(status=("bad" if isBad else "good"))
        )
        return isBad

    def getPseudoMetric(self, commit, cfg):
        commit = commit.replace('"', "")
        curThroughput = 0
        commitLogger = getCommitLogger(cfg, commit)
        isCommitCashed, cashedThroughput = self.getCommitIfCashed(commit)
        pc = Mode.CommitPath.PathCommit(
            commit,
            Mode.CommitPath.CommitState.DEFAULT
        )
        self.setOutputInfo(pc)
        self.commitPath.accept(self.traversal, pc)
        if isCommitCashed:
            logMsg = "Cashed commit - {commit}".format(commit=commit)
            self.commonLogger.info(logMsg)
            commitLogger.info(logMsg)
            curThroughput = cashedThroughput
        else:
            self.commonLogger.info("New commit: {commit}".format(
                commit=commit)
            )
            handleCommit(commit, cfg)
            output = fetchAppOutput(cfg, commit)
            commitLogger.info(output)
            foundThroughput = re.search(
                self.outPattern, output, flags=re.MULTILINE
            ).group(1)
            curThroughput = float(foundThroughput)
            self.setCommitCash(commit, curThroughput)
        return curThroughput

    def preliminaryStabilityCheck(self, commit, cfg):
        commit = commit.replace('"', "")
        curThroughput = 0

        self.commonLogger.info(
            "Preliminary check of commit: {commit}".format(
                commit=commit)
        )
        handleCommit(commit, cfg)
        throughputList = []
        dev = self.apprDev = cfg["runConfig"]["perfAppropriateDeviation"]
        for i in range(cfg["preliminaryCheckCfg"]["tryCount"]):
            output = fetchAppOutput(cfg, commit)
            foundThroughput = re.search(
                self.outPattern, output, flags=re.MULTILINE
            ).group(1)
            curThroughput = float(foundThroughput)
            throughputList.append(curThroughput)
        resStable = checkStability(throughputList, dev)
        if resStable:
            self.setCommitCash(commit, curThroughput)
        return resStable

    def setOutputInfo(self, pathCommit):
        pathCommit.perfRel = self.perfRel

    def getCommitInfo(self, commit):
        return "{ci}, perf. ratio = {d}".format(
                ci=super().getCommitInfo(commit),
                d=commit.perfRel)


class LLMBenchMode(Mode):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.perfRel = 0
        self.createCash()

    def isPerformanceBased(self):
        return True

    def prepareRun(self, list, cfg):
        super().prepareRun(list, cfg)
        sampleCommit = list[0]
        sampleCommit = sampleCommit.replace('"', "")
        self.commonLogger.info(
            "Prepare sample commit - {commit}".format(commit=sampleCommit)
        )
        commitLogger = getCommitLogger(cfg, sampleCommit)
        foundThroughput = 0
        isCommitCashed, cashedThroughput = self.getCommitIfCashed(sampleCommit)
        if isCommitCashed:
            logMsg = "Cashed commit - {commit}".format(commit=sampleCommit)
            self.commonLogger.info(logMsg)
            commitLogger.info(logMsg)
            foundThroughput = cashedThroughput
        else:
            handleCommit(sampleCommit, cfg)
            output = fetchAppOutput(cfg, sampleCommit)
            commitLogger.info(output)
            foundThroughput = re.search(
                self.outPattern, output, flags=re.MULTILINE
            ).group(1)
            self.setCommitCash(sampleCommit, float(foundThroughput))
        self.sampleThroughput = float(foundThroughput)
        return list

    def checkCfg(self, cfg):
        super().checkCfg(cfg)
        if not ("perfAppropriateDeviation" in cfg["runConfig"]):
            raise CfgError("Appropriate deviation is not configured")
        else:
            self.apprDev = cfg["runConfig"]["perfAppropriateDeviation"]
        if ("metric" in cfg["runConfig"]):
            self.outPattern = self.specifyMetric(cfg["runConfig"]["metric"])
        else:
            self.outPattern = self.specifyMetric()


    def specifyMetric(self, metric: str = "First token latency"):
        if metric in [
            "First token latency"]:
            res = r"First token latency:\s*([0-9]*[.][0-9]*)\s*ms/token"
            return res
        raise CfgError("Metric {} is not supported".format(metric))

    def preliminaryCheck(self, list, cfg):
        # # model path checking - todo is necessary ?
        # common if-degradation-exists check
        super().preliminaryCheck(list, cfg)

        # performance - specific check if results for borders are stable,
        isLeftStable = not cfg["preliminaryCheckCfg"]["leftCheck"] or\
            self.preliminaryStabilityCheck(list[0], cfg)
        isRightStable = not cfg["preliminaryCheckCfg"]["rightCheck"] or\
            self.preliminaryStabilityCheck(list[-1], cfg)
        if (not isLeftStable or not isRightStable):
            raise PreliminaryAnalysisError(
                "{lCommit} is {lStable}, {rCommit} is {rStable}".format(
                    lCommit=list[0],
                    rCommit=list[-1],
                    lStable="stable" if isLeftStable else "unstable",
                    rStable="stable" if isRightStable else "unstable"
                ),
                PreliminaryAnalysisError.PreliminaryErrType.UNSTABLE_APPLICATION
                )

    def compareCommits(self, lCommit: str, rCommit: str, cfg: map):
        leftThroughput = self.getPseudoMetric(lCommit, cfg)
        rightThroughput = self.getPseudoMetric(rCommit, cfg)
        isBad, curRel = self.traversal.numericComparator(
            leftThroughput, rightThroughput, self.apprDev
        )
        if isBad:
            self.perfRel = curRel
        curCommit = rCommit.replace('"', "")
        commitLogger = getCommitLogger(cfg, curCommit)
        commitLogger.info("Performance relation is {rel}".format(rel=curRel))
        commitLogger.info(
            "Commit is {status}".format(status=("bad" if isBad else "good"))
        )
        return isBad

    def getPseudoMetric(self, commit, cfg):
        commit = commit.replace('"', "")
        curThroughput = 0
        commitLogger = getCommitLogger(cfg, commit)
        isCommitCashed, cashedThroughput = self.getCommitIfCashed(commit)
        pc = Mode.CommitPath.PathCommit(
            commit,
            Mode.CommitPath.CommitState.DEFAULT
        )
        self.setOutputInfo(pc)
        self.commitPath.accept(self.traversal, pc)
        if isCommitCashed:
            logMsg = "Cashed commit - {commit}".format(commit=commit)
            self.commonLogger.info(logMsg)
            commitLogger.info(logMsg)
            curThroughput = cashedThroughput
        else:
            self.commonLogger.info("New commit: {commit}".format(
                commit=commit)
            )
            handleCommit(commit, cfg)
            output = fetchAppOutput(cfg, commit)
            commitLogger.info(output)
            foundThroughput = re.search(
                self.outPattern, output, flags=re.MULTILINE
            ).group(1)
            curThroughput = float(foundThroughput)
            self.setCommitCash(commit, curThroughput)
        return curThroughput

    def preliminaryStabilityCheck(self, commit, cfg):
        commit = commit.replace('"', "")
        curThroughput = 0

        self.commonLogger.info(
            "Preliminary check of commit: {commit}".format(
                commit=commit)
        )
        handleCommit(commit, cfg)
        throughputList = []
        dev = self.apprDev = cfg["runConfig"]["perfAppropriateDeviation"]
        for i in range(cfg["preliminaryCheckCfg"]["tryCount"]):
            output = fetchAppOutput(cfg, commit)
            foundThroughput = re.search(
                self.outPattern, output, flags=re.MULTILINE
            ).group(1)
            curThroughput = float(foundThroughput)
            throughputList.append(curThroughput)
        resStable = checkStability(throughputList, dev)
        if resStable:
            self.setCommitCash(commit, curThroughput)
        return resStable

    def setOutputInfo(self, pathCommit):
        pathCommit.perfRel = self.perfRel

    def getCommitInfo(self, commit):
        return "{ci}, perf. ratio = {d}".format(
                ci=super().getCommitInfo(commit),
                d=commit.perfRel)


class AccuracyCheckerMode(Mode):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.thresholdPattern = r":\s([0-9]*[.][0-9]*)%.*abs error"
        self.curMetric = None
        self.createCash()

    def prepareRun(self, list, cfg):
        super().prepareRun(list, cfg)
        sampleCommit = list[0]
        sampleCommit = sampleCommit.replace('"', "")
        self.commonLogger.info(
            "Prepare sample commit - {commit}".format(commit=sampleCommit)
        )
        commitLogger = getCommitLogger(cfg, sampleCommit)
        foundThroughput = 0
        isCommitCashed, cashedThroughput = self.getCommitIfCashed(sampleCommit)
        if isCommitCashed:
            logMsg = "Cashed commit - {commit}".format(commit=sampleCommit)
            self.commonLogger.info(logMsg)
            commitLogger.info(logMsg)
            foundThroughput = cashedThroughput
        else:
            handleCommit(sampleCommit, cfg)
            output = fetchAppOutput(cfg, sampleCommit)
            commitLogger.info(output)
            foundThroughput = re.search(
                self.thresholdPattern, output, flags=re.MULTILINE
            ).group(1)
            self.setCommitCash(sampleCommit, float(foundThroughput))
        self.sampleThroughput = float(foundThroughput)
        return list

    def compareCommits(self, lCommit: str, rCommit: str, cfg: map):
        leftMetric = self.getPseudoMetric(lCommit, cfg)
        rightMetric = self.getPseudoMetric(rCommit, cfg)
        isDiff = leftMetric != rightMetric
        if isDiff:
            self.curMetric = rightMetric
        curCommit = rCommit.replace('"', "")
        commitLogger = getCommitLogger(cfg, curCommit)
        commitLogger.info("Current accuracy is {}%".format(rightMetric))
        commitLogger.info(
            "Commit {status} from {c}".format(
                status=("differs" if isDiff else "doesn't differ"),
                c=lCommit)
        )
        return isDiff

    def getPseudoMetric(self, commit, cfg):
        commit = commit.replace('"', "")
        curThroughput = 0
        commitLogger = getCommitLogger(cfg, commit)
        isCommitCashed, cashedThroughput = self.getCommitIfCashed(commit)
        pc = Mode.CommitPath.PathCommit(
            commit,
            Mode.CommitPath.CommitState.DEFAULT
        )
        self.setOutputInfo(pc)
        self.commitPath.accept(self.traversal, pc)
        if isCommitCashed:
            logMsg = "Cashed commit - {commit}".format(commit=commit)
            self.commonLogger.info(logMsg)
            commitLogger.info(logMsg)
            curThroughput = cashedThroughput
        else:
            self.commonLogger.info("New commit: {commit}".format(
                commit=commit)
            )
            handleCommit(commit, cfg)
            output = fetchAppOutput(cfg, commit)
            commitLogger.info(output)
            foundThroughput = re.search(
                self.thresholdPattern, output, flags=re.MULTILINE
            ).group(1)
            curThroughput = float(foundThroughput)
            self.setCommitCash(commit, curThroughput)
        return curThroughput

    def setOutputInfo(self, pathCommit):
        pathCommit.metric = self.curMetric

    def getCommitInfo(self, commit):
        return "{ci}, metric = {d}".format(
                ci=super().getCommitInfo(commit),
                d=commit.metric)


class CompareBlobsMode(Mode):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.createCash()
        self.maxDiff = 0

    def prepareRun(self, list, cfg):
        # we need to exclude initial prerun-cash handling, as it may
        # lead to ignoring multiple degradations
        self.normalizeCfg(cfg)
        cfg["serviceConfig"] = {}
        # no check of prerun-cashed commits
        self.preliminaryCheck(list, cfg)
        return list

    def getPseudoMetric(self, commit, cfg):
        commit = commit.replace('"', "")
        commitLogger = getCommitLogger(cfg, commit)
        filename = ''
        isCommitCashed, cachedfileName = self.getCommitIfCashed(commit)
        pc = Mode.CommitPath.PathCommit(
            commit,
            Mode.CommitPath.CommitState.DEFAULT
        )
        self.setOutputInfo(pc)
        self.commitPath.accept(self.traversal, pc)
        if isCommitCashed:
            logMsg = "Cashed commit - {commit}".format(commit=commit)
            self.commonLogger.info(logMsg)
            commitLogger.info(logMsg)
            filename = cachedfileName
        else:
            self.commonLogger.info("New commit: {commit}".format(
                commit=commit)
            )
            handleCommit(commit, cfg)
            output = fetchAppOutput(cfg, commit)
            commitLogger.info(output)
            filename = self.setCommitCash(commit, None)
        return filename

    def compareCommits(self, lCommit: str, rCommit: str, cfg: map):
        leftBorderOutputName = self.getPseudoMetric(lCommit, cfg)
        rightBorderOutputName = self.getPseudoMetric(rCommit, cfg)
        fullLeftFileName = os.path.join(self.cachePath, leftBorderOutputName)
        fullRightName = os.path.join(self.cachePath, rightBorderOutputName)
        curMaxDiff = getBlobDiff(fullLeftFileName, fullRightName)
        isDiff = True if curMaxDiff > self.limit else False
        curCommit = rCommit
        curCommit = curCommit.replace('"', "")
        commitLogger = getCommitLogger(cfg, curCommit)
        commitLogger.info(
            "Commit {status} from {c}".format(
                status=("differs" if isDiff else "don't differ"),
                c=rCommit)
        )
        if isDiff:
            self.maxDiff = curMaxDiff
        commitLogger.info("Absolute difference is {d}".format(d=curMaxDiff))
        return isDiff

    def checkCfg(self, cfg):
        super().checkCfg(cfg)
        if not ("outputFileNamePattern" in cfg["runConfig"]):
            raise CfgError("Output pattern is not configured")
        elif not ("outputDirectory" in cfg["runConfig"]):
            raise CfgError("Output directory pattern is not configured")
        else:
            self.outFileNamePattern = cfg["runConfig"]["outputFileNamePattern"]
            self.outDir = os.path.abspath(cfg["runConfig"]["outputDirectory"])
            if "limit" in cfg["runConfig"]:
                self.limit = float(cfg["runConfig"]["limit"])
            else:
                self.limit = 0

    def setCommitCash(self, commit, valueToCache):
        isCommitCashed, _ = self.getCommitIfCashed(commit)
        newFileName = ""
        if isCommitCashed:
            raise CashError("Commit already cashed")
        else:
            fileList = os.listdir(self.outDir)
            # we look for just created output file
            for filename in fileList:
                isDump = re.search(self.outFileNamePattern, filename)
                if isDump:
                    newFileName = "{c}_{fn}".format(
                        c=getMeaningfullCommitTail(commit), fn=filename
                    )
                    shutil.move(
                        os.path.join(self.outDir, filename),
                        os.path.join(self.cachePath, newFileName)
                    )
                    break
            if filename == "":
                raise CmdError("Output file not found")
        return newFileName

    def createCash(self):
        # we use separate files instead of json cache,
        # so, we just set up path to cache folder
        # todo: handle usercache for multimodel case
        self.cachePath = getActualPath("cachePath", self.cfg)
        pass

    def getCommitIfCashed(self, commit):
        fileList = os.listdir(self.cachePath)
        curCommitPattern = "{c}_(.)*".format(c=getMeaningfullCommitTail(commit))
        for filename in fileList:
            isDump = re.search(curCommitPattern, filename)
            if isDump:
                return True, filename
        return False, None

    def setOutputInfo(self, pathCommit):
        pathCommit.diff = self.maxDiff

    def getCommitInfo(self, commit):
        return "{ci}, diff = {d}".format(
                ci=super().getCommitInfo(commit),
                d=commit.diff)
