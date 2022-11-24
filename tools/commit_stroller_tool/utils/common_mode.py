from abc import ABC
from utils.helpers import setupLogger
from utils.helpers import checkAndGetClassnameByConfig
from utils.helpers import checkAndGetSubclass
from utils.helpers import CfgError
class Mode(ABC):
    @staticmethod
    def factory(cfg):
        modeClassName = checkAndGetClassnameByConfig(cfg, "modeMap", "mode")
        cl = checkAndGetSubclass(modeClassName, Mode)
        return cl(cfg)
    def __init__(self, cfg) -> None:
        self.checkCfg(cfg)
        traversalClassName = checkAndGetClassnameByConfig(cfg, "traversalMap", "traversal")
        traversalClass = checkAndGetSubclass(traversalClassName, self.Traversal)
        self.traversal = traversalClass(self)
        self.cfg = cfg
        self.commonLogger = setupLogger(
            'commonLogger',
            '{logPath}/common_log.log'.format(logPath=cfg["commonConfig"]["logPath"]))
    def createCash(self):
        raise NotImplementedError("createCash() not implemented")
    def getCommitIfCashed(self, commit):
        raise NotImplementedError("getCommitIfCashed() not implemented")
    def setCommitCash(self, commit, valueToCache):
        raise NotImplementedError("setCommitCash() not implemented")
    def checkCfg(self, cfg):
        if not("traversal" in cfg["specialConfig"]):
            raise CfgError("traversal is not configured")
    def isBadVersion(commit, cfg):
        raise NotImplementedError("isBadVersion() is not implemented")
    def run(self, i1, i2, list, cfg) -> int:
        cfg["serviceConfig"] = {} # prepare service data
        return self.traversal.bypass(i1, i2, list, cfg, self.isBadVersion)

    
    class Traversal(ABC):
        def bypass(self, i1, i2, list, cfg, isBadVersion) -> int:
            raise NotImplementedError()
        def __init__(self, mode) -> None:
            self.mode = mode

    class FirstBadVersion(Traversal):
        def __init__(self, mode) -> None:
            super().__init__(mode)
        def bypass(self, i1, i2, list, cfg, isBadVersion) -> int:
            # check cfg if necessary
            noCleanInterval = cfg["commonConfig"]["noCleanInterval"]
            cfg["serviceConfig"]["trySkipClean"] = (True 
                if i2 - i1 < noCleanInterval
                else False)
            self.mode.commonLogger.info("Check interval {i1}..{i2}".format(i1=i1, i2=i2))
            self.mode.commonLogger.info("Check commits {c1}..{c2}".format(c1=list[i1], c2=list[i2]))
            #
            if (i1 + 1 >= i2):
                return i1 if isBadVersion(list[i1], cfg) else i2
            mid = (int)((i1 + i2) / 2)
            if (isBadVersion(list[mid], cfg)):
                return self.bypass(i1, mid, list, cfg, isBadVersion)
            else:
                return self.bypass(mid, i2, list, cfg, isBadVersion)
