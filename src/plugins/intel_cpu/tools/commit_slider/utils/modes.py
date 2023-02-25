import os
from utils.helpers import fetchAppOutput, getActualPath
from utils.helpers import getMeaningfullCommitTail
from utils.helpers import handleCommit, runCommandList, getBlobDiff
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

    def checkIfBordersDiffer(self, i1, i2, list, cfg):
        isLeftBorderFailed = False
        if i1 != 0 or cfg["checkIfBordersDiffer"]:
            isLeftBorderFailed = self.isBadVersion(list[i1], cfg)

        isRightBorderGood = not self.isBadVersion(list[i2], cfg)
        rightCommit = list[i2]
        rightCommit = rightCommit.replace('"', "")
        commitLogger = getCommitLogger(cfg, rightCommit)
        commitLogger.info(
            "Commit {c} is {status}".format(
                status=("good" if isRightBorderGood else "bad"),
                c=list[i2])
        )
        return isLeftBorderFailed == isRightBorderGood

    def isBadVersion(self, commit, cfg):
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
            checkOut = fetchAppOutput(cfg)
            commitLogger.info(checkOut)
            self.setCommitCash(commit, checkOut)
        stopPattern = cfg["runConfig"]["stopPattern"]
        isFound = re.search(stopPattern, checkOut)
        return isFound


class BenchmarkAppPerformanceMode(Mode):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.outPattern = "Throughput:\s*([0-9]*[.][0-9]*)\s*FPS"
        self.perfRel = 0
        self.createCash()

    def prepareRun(self, i1, i2, list, cfg):
        super().prepareRun(i1, i2, list, cfg)
        sampleCommit = list[i1]
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
            runCommandList(sampleCommit, cfg, enforceClean=True)
            output = fetchAppOutput(cfg)
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

    def checkIfBordersDiffer(self, i1, i2, list, cfg):
        leftThroughput = self.getThroughputByCommit(list[i1], cfg)
        rightCommit = list[i2]
        rightThroughput = self.getThroughputByCommit(rightCommit, cfg)
        curRel = rightThroughput / leftThroughput
        isBad = not ((1 - curRel) < self.apprDev)
        if isBad:
            self.perfRel = curRel
        rightCommit = rightCommit.replace('"', "")
        commitLogger = getCommitLogger(cfg, rightCommit)
        commitLogger.info("Performance relation is {rel}".format(rel=curRel))
        commitLogger.info(
            "Commit is {status}".format(status=("bad" if isBad else "good"))
        )
        return isBad

    def getThroughputByCommit(self, commit, cfg):
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
            output = fetchAppOutput(cfg)
            foundThroughput = re.search(
                self.outPattern, output, flags=re.MULTILINE
            ).group(1)
            curThroughput = float(foundThroughput)
            commitLogger.info(output)
            self.setCommitCash(commit, curThroughput)
        return curThroughput

    def setOutputInfo(self, pathCommit):
        pathCommit.perfRel = self.perfRel

    def getResult(self):
        for pathCommit in self.commitPath.getList():
            print("Break commit: {c}, perf. ratio = {d}".format(
                c=self.commitList[pathCommit.id],
                d=pathCommit.perfRel)
            )


class CompareBlobsMode(Mode):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.createCash()
        self.maxDiff = 0

    def getOutNameByCommit(self, commit, cfg):
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
            runCommandList(commit, cfg, enforceClean=True)
            output = fetchAppOutput(cfg)
            commitLogger.info(output)
            filename = self.setCommitCash(commit, None)
        return filename

    def checkIfBordersDiffer(self, i1, i2, list, cfg):
        leftBorderOutputName = self.getOutNameByCommit(list[i1], cfg)
        rightBorderOutputName = self.getOutNameByCommit(list[i2], cfg)
        fullLeftFileName = os.path.join(self.cachePath, leftBorderOutputName)
        fullRightName = os.path.join(self.cachePath, rightBorderOutputName)
        curMaxDiff = getBlobDiff(fullLeftFileName, fullRightName)
        isDiff = True if curMaxDiff > self.limit else False
        rightCommit = list[i2]
        rightCommit = rightCommit.replace('"', "")
        commitLogger = getCommitLogger(cfg, rightCommit)
        commitLogger.info(
            "Commit {status} from {c}".format(
                status=("differs" if isDiff else "doesn't differ"),
                c=list[i2])
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

    def getResult(self):
        for pathcommit in self.commitPath.getList():
            print("Break commit: {c}, diff = {d}".format(
                c=self.commitList[pathcommit.id],
                d=pathcommit.diff)
            )
