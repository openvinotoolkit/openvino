# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from utils.helpers import CfgManager, fetchAppOutput, getActualPath
from utils.helpers import getMeaningfullCommitTail, extractModelPath
from utils.helpers import handleCommit, getBlobDiff, applySubstitutionRules, simpleSubstitute
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
        self.onlyMsg = 'onlyMsg' in cfg['runConfig'] and cfg['runConfig']['onlyMsg']
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
        if not self.onlyMsg:
            print("current commit: {}".format(commit))
            print(checkOut)
        return

    def printResult(self):
        # if CS launched with template we use custom representation
        # if not, as default we print msg attribute
        print(self.msg)
        self.outLogger.info(self.msg)


class CrossCheckMode(Mode):
    def __init__(self, cfg):
        super().__init__(cfg)

    def checkCfg(self, cfg):
        self.par_1 = cfg["runConfig"]["par_1"]
        self.par_2 = cfg["runConfig"]["par_2"]
        # todo: extend for another metrics
        self.outPattern = r'{spec}:\s*([0-9]*[.][0-9]*)\s*{measure}'.format(
                spec='Throughput', measure='FPS')
        super().checkCfg(cfg)

    def getPseudoMetric(self, commit, cfg):
        commit = commit.replace('"', "")
        commitLogger = getCommitLogger(cfg, commit)
        self.commonLogger.info("New commit: {commit}".format(
            commit=commit)
        )
        handleCommit(commit, cfg)
        fullOutput = ""
        simpleSubstitute(cfg, "actualPar", "$.runConfig.par_1", "$.appCmd")

        # run first app
        checkOut = fetchAppOutput(cfg, commit)
        foundThroughput = re.search(
                self.outPattern, checkOut, flags=re.MULTILINE
            ).group(1)
        self.firstThroughput = foundThroughput
        self.firstModel = cfg['appCmd']
        fullOutput = checkOut
        simpleSubstitute(cfg, "actualPar", "$.runConfig.par_2", "$.appCmd")

        # run second app
        checkOut = fetchAppOutput(cfg, commit)
        foundThroughput = re.search(
                self.outPattern, checkOut, flags=re.MULTILINE
            ).group(1)
        self.secondThroughput = foundThroughput
        self.secondModel = cfg['appCmd']
        pc = Mode.CommitPath.PathCommit(
            commit,
            Mode.CommitPath.CommitState.DEFAULT
        )
        self.setOutputInfo(pc)
        self.commitPath.accept(self.traversal, pc)

        fullOutput = fullOutput + checkOut

        commitLogger.info(fullOutput)

    def printResult(self):
        if self.cfg['template'] == 'common_template':
            for pathcommit in self.commitPath.getList():
                commitInfo = self.getCommitInfo(pathcommit)
                print(commitInfo)
                self.outLogger.info(commitInfo)
        else:
            from utils.templates.common import CommonTemplate
            tmpl = CommonTemplate.getTemplate(self.cfg['template'])
            tmpl.printResult(self.commitPath, self.outLogger, self.getCommitInfo)


    def setOutputInfo(self, pathCommit):
        pathCommit.firstThroughput = self.firstThroughput
        pathCommit.firstModel = self.firstModel
        pathCommit.secondThroughput = self.secondThroughput
        pathCommit.secondModel = self.secondModel

    def getCommitInfo(self, commit):
        return "{hash}, throughput_1 = {t1}, throughput_2 = {t2}".format(
            hash=commit.cHash,
            t1=commit.firstThroughput,
            t2=commit.secondThroughput)


class ModelCompilationMode(Mode):
    def __init__(self, cfg):
        super().__init__(cfg)

    def checkCfg(self, cfg):
        self.par_1 = cfg["runConfig"]["par_1"]
        self.par_2 = cfg["runConfig"]["par_2"]
        # todo: extend for another metrics
        self.outPattern = r'{spec}\s*([0-9]*[.][0-9]*)\s*{measure}'.format(
                spec='Compile model took', measure='ms')
        super().checkCfg(cfg)

    def getPseudoMetric(self, commit, cfg):
        commit = commit.replace('"', "")
        commitLogger = getCommitLogger(cfg, commit)
        self.commonLogger.info("New commit: {commit}".format(
            commit=commit)
        )
        handleCommit(commit, cfg)
        fullOutput = ""
        simpleSubstitute(cfg, "actualPar", "$.runConfig.par_1", "$.appCmd")

        # run first app
        checkOut = fetchAppOutput(cfg, commit)
        foundThroughput = re.search(
                self.outPattern, checkOut, flags=re.MULTILINE
            ).group(1)
        self.firstThroughput = foundThroughput
        self.firstModel = cfg['appCmd']
        fullOutput = checkOut
        simpleSubstitute(cfg, "actualPar", "$.runConfig.par_2", "$.appCmd")

        # run second app
        checkOut = fetchAppOutput(cfg, commit)
        foundThroughput = re.search(
                self.outPattern, checkOut, flags=re.MULTILINE
            ).group(1)
        self.secondThroughput = foundThroughput
        self.secondModel = cfg['appCmd']
        pc = Mode.CommitPath.PathCommit(
            commit,
            Mode.CommitPath.CommitState.DEFAULT
        )
        self.setOutputInfo(pc)
        self.commitPath.accept(self.traversal, pc)

        fullOutput = fullOutput + checkOut

        commitLogger.info(fullOutput)

    def printResult(self):
        if self.cfg['template'] == 'common_template':
            for pathcommit in self.commitPath.getList():
                commitInfo = self.getCommitInfo(pathcommit)
                print(commitInfo)
                self.outLogger.info(commitInfo)
        else:
            from utils.templates.common import CommonTemplate
            tmpl = CommonTemplate.getTemplate(self.cfg['template'])
            tmpl.printResult(self.commitPath, self.outLogger, self.getCommitInfo)


    def setOutputInfo(self, pathCommit):
        pathCommit.firstThroughput = self.firstThroughput
        pathCommit.firstModel = self.firstModel
        pathCommit.secondThroughput = self.secondThroughput
        pathCommit.secondModel = self.secondModel

    def getCommitInfo(self, commit):
        return "{hash}, throughput_1 = {t1}, throughput_2 = {t2}".format(
            hash=commit.cHash,
            t1=commit.firstThroughput,
            t2=commit.secondThroughput)


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
         # option of automatic matching of comparing blobs i.e.
         # blob_for_node_1_commit_1.ieb <--> blob_for_node_1_commit_2.ieb
         # blob_for_node_2_commit_1.ieb <--> blob_for_node_2_commit_2.ieb

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

    def compareSingleBlobPair(self, lName, rName, commit, cfg):
        curMaxDiff = getBlobDiff(lName, rName)
        isDiff = True if curMaxDiff > self.limit else False
        curCommit = commit
        curCommit = curCommit.replace('"', "")
        commitLogger = getCommitLogger(cfg, curCommit)
        commitLogger.info(
            "Commit {status} from {c}".format(
                status=("differs" if isDiff else "don't differ"),
                c=commit))
        commitLogger.info("Absolute difference is {d}".format(d=curMaxDiff))
        if isDiff:
            self.maxDiff = curMaxDiff
        return isDiff

    def compareCommits(self, lCommit: str, rCommit: str, cfg: map):
        leftOutput = self.getPseudoMetric(lCommit, cfg)
        rightOutput = self.getPseudoMetric(rCommit, cfg)

        isDiff = False
        matchingPairList = []
        if self.autoMatch:
            # auto matching of patterns
            from utils.helpers import getBlobMatch
            matchingPairList = getBlobMatch(leftOutput, rightOutput, lCommit, rCommit)
        else:
            # single specified blob pattern
            matchingPairList = [[leftOutput, rightOutput]]
        for matchPair in matchingPairList:
            fullLeftFileName = os.path.join(self.cachePath, matchPair[0])
            fullRightFileName = os.path.join(self.cachePath, matchPair[1])
            isDiff = isDiff or self.compareSingleBlobPair(fullLeftFileName, fullRightFileName, rCommit, cfg)
        return isDiff

    def checkCfg(self, cfg):
        super().checkCfg(cfg)
        isAutoMatch = "autoMatch" in cfg["runConfig"] and \
            cfg["runConfig"]["autoMatch"] is True
        if not ("outputFileNamePattern" in cfg["runConfig"]) and not isAutoMatch:
            raise CfgError("Output pattern OR automatch is not configured")
        elif not ("outputDirectory" in cfg["runConfig"]) and not isAutoMatch:
            raise CfgError("Output directory pattern is not configured")
        else:
            if isAutoMatch:
                self.autoMatch = True
                cfg['defaultTmpDir'] = CfgManager.singlestepStrFormat(cfg["defaultTmpDir"], "workPath", cfg['workPath'])
                os.makedirs(cfg['defaultTmpDir'], exist_ok=True)
                self.outDir = os.path.abspath(cfg["defaultTmpDir"])
            else:
                self.autoMatch = False
                self.outFileNamePattern = cfg["runConfig"]["outputFileNamePattern"]
                self.outDir = os.path.abspath(cfg["runConfig"]["outputDirectory"])
            if "limit" in cfg["runConfig"]:
                self.limit = float(cfg["runConfig"]["limit"])
            else:
                self.limit = 0

    def setCommitCash(self, commit, valueToCache):
        isCommitCashed, _ = self.getCommitIfCashed(commit)
        newFileName = [] if self.autoMatch else ""
        if isCommitCashed:
            raise CashError("Commit already cashed")
        else:
            fileList = os.listdir(self.outDir)
            # we look for just created output file
            for filename in fileList:
                isDump = True if self.autoMatch else re.search(self.outFileNamePattern, filename)
                if isDump:
                    curFileName = "{c}_{fn}".format(
                        c=getMeaningfullCommitTail(commit), fn=filename
                    )
                    shutil.move(
                        os.path.join(self.outDir, filename),
                        os.path.join(self.cachePath, curFileName)
                    )
                    if self.autoMatch:
                        newFileName.append(curFileName)
                    else:
                        newFileName = curFileName
                        break
        return newFileName

    def createCash(self):
        # we use separate files instead of json cache,
        # so, we just set up path to cache folder
        # todo: handle usercache for multimodel case
        self.cachePath = getActualPath("cachePath", self.cfg)

    def getCommitIfCashed(self, commit):
        fileList = os.listdir(self.cachePath)
        curCommitPattern = "{c}_(.)*".format(c=getMeaningfullCommitTail(commit))
        
        cashedFileList = []
        for filename in fileList:
            isDump = re.search(curCommitPattern, filename)
            if isDump:
                if self.autoMatch:
                    cashedFileList.append(filename)
                else:
                    return True, filename
        if self.autoMatch:
            return not (not cashedFileList), cashedFileList
        else:
            return False, None

    def setOutputInfo(self, pathCommit):
        pathCommit.diff = self.maxDiff

    def getCommitInfo(self, commit):
        return "{ci}, diff = {d}".format(
                ci=super().getCommitInfo(commit),
                d=commit.diff)
