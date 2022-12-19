import os
from utils.helpers import fetchAppOutput, getActualPath
from utils.helpers import getMeaningfilCommitTail
from utils.helpers import handleCommit, runCommandList
from utils.helpers import getCommitLogger, CashError, CfgError, CmdError
import re
import shutil
from utils.common_mode import Mode


class CheckOutputMode(Mode):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.createCash()

    def prepareRun(self, i1, i2, list, cfg):
        super().prepareRun(i1, i2, list, cfg)

    def checkCfg(self, cfg):
        super().checkCfg(cfg)
        if not ("stopPattern" in cfg["runConfig"]):
            raise CfgError("stopPattern is not configured")

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
        self.createCash()

    def prepareRun(self, i1, i2, list, cfg):
        super().prepareRun(i1, i2, list, cfg)
        sampleCommit = list[i1]
        sampleCommit = sampleCommit.replace('"', "")
        self.commonLogger.info(
            "Prepare sample commit - {commit}".format(commit=sampleCommit)
        )
        commitLogger = getCommitLogger(cfg, sampleCommit)
        cfg["trySkipClean"] = False
        foundThroughput = 0
        isCommitCashed, cashedThroughput = self.getCommitIfCashed(sampleCommit)
        if isCommitCashed:
            logMsg = "Cashed commit - {commit}".format(commit=sampleCommit)
            self.commonLogger.info(logMsg)
            commitLogger.info(logMsg)
            foundThroughput = cashedThroughput
        else:
            runCommandList(sampleCommit, cfg)
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

    def isBadVersion(self, commit, cfg):
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
        curRel = curThroughput / self.sampleThroughput
        isBad = not (abs(1 - curRel) < self.apprDev)
        commitLogger.info("Performance relation is {rel}".format(rel=curRel))
        commitLogger.info(
            "Commit is {status}".format(status=("bad" if isBad else "good"))
        )
        return not (abs(1 - curRel) < self.apprDev)


class CompareBlobsMode(Mode):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sampleFileName = "undefined"
        self.createCash()

    def prepareRun(self, i1, i2, list, cfg):
        super().prepareRun(i1, i2, list, cfg)
        sampleCommit = list[i1]
        sampleCommit = sampleCommit.replace('"', "")
        self.commonLogger.info(
            "Prepare sample commit - {commit}".format(commit=sampleCommit)
        )
        commitLogger = getCommitLogger(cfg, sampleCommit)
        cfg["trySkipClean"] = False
        isCommitCashed, cachedfileName = self.getCommitIfCashed(sampleCommit)
        if isCommitCashed:
            logMsg = "Cashed commit - {commit}".format(commit=sampleCommit)
            self.commonLogger.info(logMsg)
            commitLogger.info(logMsg)
            self.sampleFileName = cachedfileName
        else:
            runCommandList(sampleCommit, cfg)
            output = fetchAppOutput(cfg)
            commitLogger.info(output)
            self.sampleFileName = self.setCommitCash(sampleCommit, None)

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

    def isBadVersion(self, commit, cfg):
        commit = commit.replace('"', "")
        commitLogger = getCommitLogger(cfg, commit)
        isCommitCashed, fileName = self.getCommitIfCashed(commit)
        if isCommitCashed:
            logMsg = "Cashed commit - {commit}".format(commit=commit)
            self.commonLogger.info(logMsg)
            commitLogger.info(logMsg)
        else:
            self.commonLogger.info("New commit: {commit}".format(
                commit=commit)
            )
            handleCommit(commit, cfg)
            output = fetchAppOutput(cfg)
            commitLogger.info(output)
            fileName = self.setCommitCash(commit, None)
        sampleFileName = self.sampleFileName
        fullSampleFileName = os.path.join(self.cachePath, sampleFileName)
        with open(fullSampleFileName) as sampleFile:
            sampleContent = sampleFile.readlines()
        fullFileName = os.path.join(self.cachePath, fileName)
        with open(fullFileName) as file:
            content = file.readlines()
        # ignore first line with memory address
        i = -1
        maxDiff = 0
        for sampleLine in sampleContent:
            i = i + 1
            if i >= len(sampleContent):
                break
            line = content[i]
            sampleVal = 0
            val = 0
            try:
                sampleVal = float(sampleLine)
                val = float(line)
            except ValueError:
                continue
            if val != sampleVal:
                maxDiff = max(maxDiff, abs(val - sampleVal))
        isBad = maxDiff > self.limit
        commitLogger.info(
            "Commit is {status}".format(status=("bad" if isBad else "good"))
        )
        commitLogger.info("Absolute difference is {diff}".format(diff=maxDiff))
        return isBad

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
                        c=getMeaningfilCommitTail(commit), fn=filename
                    )
                    shutil.copyfile(
                        os.path.join(self.outDir, filename),
                        os.path.join(self.cachePath, newFileName),
                        follow_symlinks=True,
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
        curCommitPattern = "{c}_(.)*".format(c=getMeaningfilCommitTail(commit))
        for filename in fileList:
            isDump = re.search(curCommitPattern, filename)
            if isDump:
                return True, filename
        return False, None
