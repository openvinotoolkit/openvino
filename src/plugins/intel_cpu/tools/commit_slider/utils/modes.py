import os
from utils.helpers import fetchAppOutput, getActualPath
from utils.helpers import getMeaningfullCommitTail
from utils.helpers import handleCommit, getBlobDiff
from utils.helpers import getCommitLogger, CashError, CfgError, CmdError
import re
import shutil
from utils.common_mode import Mode


class CheckOutputMode(Mode):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.createCash()

    def checkCfg(self, cfg):
        super().checkCfg(cfg)
        if not ("stopPattern" in cfg["runConfig"]):
            raise CfgError("stopPattern is not configured")

    def compareCommits(self, lCommit, rCommit, list, cfg):
        isLeftBorderFailed = self.getPseudoMetric(lCommit, cfg)
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
        self.outPattern = "Throughput:\s*([0-9]*[.][0-9]*)\s*FPS"
        self.perfRel = 0
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
                self.outPattern, output, flags=re.MULTILINE
            ).group(1)
            self.setCommitCash(sampleCommit, float(foundThroughput))
        self.sampleThroughput = float(foundThroughput)

    def checkCfg(self, cfg):
        super().checkCfg(cfg)
        if not ("perfAppropriateDeviation" in cfg["runConfig"]):
            raise CfgError("Appropriate deviation is not configured")
        else:
            self.apprDev = cfg["runConfig"]["perfAppropriateDeviation"]

    def compareCommits(self, lCommit, rCommit, list, cfg):
        leftThroughput = self.getPseudoMetric(lCommit, cfg)
        rightThroughput = self.getPseudoMetric(rCommit, cfg)
        curRel = rightThroughput / leftThroughput
        isBad = not ((1 - curRel) < self.apprDev)
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

    def setOutputInfo(self, pathCommit):
        pathCommit.perfRel = self.perfRel

    def getCommitInfo(self, commit):
        return "{ci}, perf. ratio = {d}".format(
                ci=super().getCommitInfo(commit),
                d=commit.perfRel)


class CompareBlobsMode(Mode):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.createCash()
        self.maxDiff = 0

    def getPseudoMetric(self, commit, cfg):
        commit = commit.replace('"', "")
        commitLogger = getCommitLogger(cfg, commit)
        filename = ''
        isCommitCashed, cachedfileName = self.getCommitIfCashed(commit)
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

    def compareCommits(self, lCommit, rCommit, list, cfg):
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
