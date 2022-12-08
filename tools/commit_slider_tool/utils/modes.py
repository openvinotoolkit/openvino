import os
from utils.helpers import fetchAppOutput, handleCommit, runCommandList
from utils.helpers import getCommitLogger
from utils.helpers import CashError
from utils.helpers import CfgError
import subprocess
import re
import json
from utils.common_mode import Mode

class CheckOutputMode(Mode):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.createCash()
    
    def prepareRun(self, i1, i2, list, cfg):
        super().prepareRun(i1, i2, list, cfg)

    def createCash(self):
        wp = self.cfg["workPath"]
        cp = self.cfg["cachePath"]
        cp = cp.format(workPath=wp)
        if not os.path.exists(cp):
            os.makedirs(cp)
        self.cachePath = os.path.join(cp, 'check_output_cache.json')
        initCacheMap = {}
        try:
            cacheDump = open(self.cachePath, 'r+')
            if self.cfg["clearCache"]:
                cacheDump.truncate(0)
                json.dump(initCacheMap, cacheDump)
            else:
                try:
                    json.load(cacheDump)
                except json.decoder.JSONDecodeError:
                    json.dump(initCacheMap, cacheDump)
        except FileNotFoundError:
            cacheDump = open(self.cachePath, 'w')
            json.dump(initCacheMap, cacheDump)
        cacheDump.close()

    def checkCfg(self, cfg):
        super().checkCfg(cfg)
        if not("stopPattern" in cfg["runConfig"]):
            raise CfgError("stopPattern is not configured")

    def getCommitIfCashed(self, commit):
        with open(self.cachePath, 'r') as cacheDump:
            cacheData = json.load(cacheDump)
            cacheDump.close()
            if commit in cacheData:
                return True, cacheData[commit]
            else:
                return False, None

    def setCommitCash(self, commit, valueToCache):
        isCommitCashed, _ = self.getCommitIfCashed(commit)
        if (isCommitCashed):
            raise CashError("Commit already cashed")
        else:
            with open(self.cachePath, 'r+', encoding='utf-8') as cacheDump:
                cacheData = json.load(cacheDump)
                cacheData[commit] = valueToCache
                cacheDump.seek(0)
                json.dump(cacheData, cacheDump, indent = 4)
                cacheDump.truncate()
                cacheDump.close()

    def isBadVersion(self, commit, cfg):
        commit = commit.replace('"', '')
        checkOut = ""
        commitLogger = getCommitLogger(cfg, commit)
        isCommitCashed, cashedOutput = self.getCommitIfCashed(commit)
        if (isCommitCashed):
            logMsg = "Cashed commit - {commit}".format(commit = commit)
            self.commonLogger.info(logMsg)
            commitLogger.info(logMsg)
            checkOut = cashedOutput
        else:
            self.commonLogger.info("New commit - {commit}".format(commit = commit))
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
        self.outPattern = 'Throughput: ([0-9]*[.][0-9]*) FPS'
        self.createCash()

    def prepareRun(self, i1, i2, list, cfg):
        super().prepareRun(i1, i2, list, cfg)
        sampleCommit = list[i1]
        sampleCommit = sampleCommit.replace('"', '')
        self.commonLogger.info("Prepare sample commit - {commit}".format(commit = sampleCommit))
        commitLogger = getCommitLogger(cfg, sampleCommit)
        cfg["trySkipClean"] = False
        foundThroughput = 0
        isCommitCashed, cashedThroughput = self.getCommitIfCashed(sampleCommit)
        if (isCommitCashed):
            logMsg = "Cashed commit - {commit}".format(commit = sampleCommit)
            self.commonLogger.info(logMsg)
            commitLogger.info(logMsg)
            foundThroughput = cashedThroughput
        else:
            runCommandList(sampleCommit, cfg)
            output = fetchAppOutput(cfg)
            commitLogger.info(output)
            foundThroughput = re.search(self.outPattern, output, flags=re.MULTILINE).group(1)
            self.setCommitCash(sampleCommit, float(foundThroughput))
        self.sampleThroughput = float(foundThroughput)

    def checkCfg(self, cfg):
        super().checkCfg(cfg)
        if not("perfAppropriateDeviation" in cfg["runConfig"]):
            raise CfgError("Appropriate deviation is not configured")
        else:
            self.apprDev = cfg["runConfig"]["perfAppropriateDeviation"]

    def isBadVersion(self, commit, cfg):
        commit = commit.replace('"', '')
        curThroughput = 0
        commitLogger = getCommitLogger(cfg, commit)
        isCommitCashed, cashedThroughput = self.getCommitIfCashed(commit)
        if (isCommitCashed):
            logMsg = "Cashed commit - {commit}".format(commit = commit)
            self.commonLogger.info(logMsg)
            commitLogger.info(logMsg)
            curThroughput = cashedThroughput
        else:
            self.commonLogger.info("New commit - {commit}".format(commit = commit))
            handleCommit(commit, cfg)
            output = fetchAppOutput(cfg)
            foundThroughput = re.search(self.outPattern, output, flags=re.MULTILINE).group(1)
            curThroughput = float(foundThroughput)
            commitLogger.info(output)
            self.setCommitCash(commit, curThroughput)
        curRel = curThroughput / self.sampleThroughput
        isBad = not(abs(1 - curRel) < self.apprDev)
        commitLogger.info("Performance relation is {rel}".format(rel=curRel))
        commitLogger.info("Commit is {status}".format(status=('bad' if isBad else 'good')))
        return not(abs(1 - curRel) < self.apprDev)
