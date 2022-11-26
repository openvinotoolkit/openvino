from utils.helpers import handleCommit
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

    def createCash(self):
        self.cachePath = self.cfg["commonConfig"]["cachePath"] + '/check_output_cache.json'
        with open(self.cachePath, 'r') as cacheDump:
            try:
                json.load(cacheDump)
            except json.decoder.JSONDecodeError:
                initCacheMap = {}
                json.dump(initCacheMap, cacheDump)
        cacheDump.close()

    def checkCfg(self, cfg):
        super().checkCfg(cfg)
        if not("stopPattern" in cfg["specialConfig"]):
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
            appCmd = cfg["commonConfig"]["appCmd"]
            appPath = cfg["commonConfig"]["appPath"]
            p = subprocess.Popen(appCmd.split(), cwd=appPath, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            checkOut, err = p.communicate()
            commitLogger.info(checkOut)
            checkOut = checkOut.decode('utf-8')
            self.setCommitCash(commit, checkOut)
        stopPattern = cfg["specialConfig"]["stopPattern"]
        isFound = re.search(stopPattern, checkOut)
        return isFound
