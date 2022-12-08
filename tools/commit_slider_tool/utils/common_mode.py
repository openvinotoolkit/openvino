from abc import ABC
import utils.helpers as util
import json
import os

class Mode(ABC):
    @staticmethod
    def factory(cfg):
        modeClassName = util.checkAndGetClassnameByConfig(cfg, "modeMap", "mode")
        cl = util.checkAndGetSubclass(modeClassName, Mode)
        return cl(cfg)

    def __init__(self, cfg) -> None:
        self.checkCfg(cfg)
        traversalClassName = util.checkAndGetClassnameByConfig(cfg, "traversalMap", "traversal")
        traversalClass = util.checkAndGetSubclass(traversalClassName, self.Traversal)
        self.traversal = traversalClass(self)
        self.cfg = cfg
        logPath = util.getActualPath("logPath", cfg)
        self.commonLogger = util.setupLogger('commonLogger', logPath, 'common_log.log')

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
            raise util.CashError("Commit already cashed")
        else:
            with open(self.cachePath, 'r+', encoding='utf-8') as cacheDump:
                cacheData = json.load(cacheDump)
                cacheData[commit] = valueToCache
                cacheDump.seek(0)
                json.dump(cacheData, cacheDump, indent = 4)
                cacheDump.truncate()
                cacheDump.close()

    def checkCfg(self, cfg):
        if not("traversal" in cfg["runConfig"]):
            raise util.CfgError("traversal is not configured")

    def isBadVersion(commit, cfg):
        raise NotImplementedError("isBadVersion() is not implemented")

    def prepareRun(self, i1, i2, list, cfg):
        cfg["serviceConfig"] = {}

    def run(self, i1, i2, list, cfg) -> int:
        self.prepareRun(i1, i2, list, cfg)
        self.commitList = list
        self.endCommit = self.traversal.bypass(i1, i2, list, cfg, self.isBadVersion)
        util.returnToActualVersion(self.cfg)

    def getResult(self):
        # override if you need more than one-found-commit representation
        print ("Commit found: {c}".format(c=self.commitList[self.endCommit]))
    
    class Traversal(ABC):
        def bypass(self, i1, i2, list, cfg, isBadVersion) -> int:
            raise NotImplementedError()
        def prepBypass(self, i1, i2, list, cfg):
            noCleanInterval = cfg["noCleanInterval"]
            cfg["serviceConfig"]["skipCleanInterval"] = (i2 - i1 < noCleanInterval)
            self.mode.commonLogger.info("Check interval {i1}..{i2}".format(i1=i1, i2=i2))
            self.mode.commonLogger.info("Check commits {c1}..{c2}".format(c1=list[i1], c2=list[i2]))
        def __init__(self, mode) -> None:
            self.mode = mode

    class FirstFailedVersion(Traversal):
        def __init__(self, mode) -> None:
            super().__init__(mode)
        def bypass(self, i1, i2, list, cfg, isBadVersion) -> int:
            self.prepBypass(i1, i2, list, cfg)
            if (i1 + 1 >= i2):
                return i1 if isBadVersion(list[i1], cfg) else i2
            mid = (int)((i1 + i2) / 2)
            if (isBadVersion(list[mid], cfg)):
                return self.bypass(i1, mid, list, cfg, isBadVersion)
            else:
                return self.bypass(mid, i2, list, cfg, isBadVersion)
    
    class FirstFixedVersion(Traversal):
        def __init__(self, mode) -> None:
            super().__init__(mode)
        def bypass(self, i1, i2, list, cfg, isBadVersion) -> int:
            self.prepBypass(i1, i2, list, cfg)
            if (i1 + 1 >= i2):
                return i2 if isBadVersion(list[i1], cfg) else i1
            mid = (int)((i1 + i2) / 2)
            if (isBadVersion(list[mid], cfg)):
                return self.bypass(mid, i2, list, cfg, isBadVersion)
            else:
                return self.bypass(i1, mid, list, cfg, isBadVersion)
