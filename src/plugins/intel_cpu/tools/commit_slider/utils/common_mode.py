from abc import ABC
import utils.helpers as util
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

    def setCommitCash(self, commit, valueToCache):
        isCommitCashed, _ = self.getCommitIfCashed(commit)
        if isCommitCashed:
            raise util.CashError("Commit already cashed")
        else:
            with open(self.cachePath, "r+", encoding="utf-8") as cacheDump:
                cacheData = json.load(cacheDump)
                cacheData[commit] = valueToCache
                cacheDump.seek(0)
                json.dump(cacheData, cacheDump, indent=4)
                cacheDump.truncate()
                cacheDump.close()

    def checkCfg(self, cfg):
        if not ("traversal" in cfg["runConfig"]):
            raise util.CfgError("traversal is not configured")

    def prepareRun(self, i1, i2, list, cfg):
        cfg["serviceConfig"] = {}
        if cfg["checkIfBordersDiffer"] and not self.checkIfListBordersDiffer(
                list, cfg):
            raise util.RepoError("Borders {i1} and {i2} doesn't differ".format(
                i1=i1, i2=i2))
        self.commitList = list

    def postRun(self, list):
        util.returnToActualVersion(self.cfg)
        if "printCSV" in self.cfg and self.cfg["printCSV"]:
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

    def run(self, i1, i2, list, cfg) -> int:
        self.prepareRun(i1, i2, list, cfg)
        self.traversal.bypass(
            i1, i2, list, cfg, self.commitPath
        )
        self.postRun(list)

    def setOutputInfo(self, pathCommit):
        # override if you need more details in output representation
        pass

    def getResult(self):
        # override if you need more details in output representation
        for pathcommit in self.commitPath.getList():
            print("Break commit: {c}".format(
                c=self.commitList[pathcommit.id])
            )

    def checkIfBordersDiffer(self, i1, i2, list, cfg):
        raise NotImplementedError("checkIfBordersDiffer() is not implemented")

    def checkIfListBordersDiffer(self, list, cfg):
        return self.checkIfBordersDiffer(0, len(list) - 1, list, cfg)

    class CommitPath:

        def __init__(self):
            self.commitList = []

        def accept(self, traversal, commitToReport) -> None:
            traversal.visit(self, commitToReport)

        class CommitState(Enum):
            BREAK = 1
            SKIPPED = 2

        class PathCommit:
            def __init__(self, id, state):
                self.id = id
                self.state = state

        def append(self, commit):
            self.commitList.append(commit)

        def pop(self):
            return self.commitList.pop(0)

        def getList(self):
            return self.commitList

    class Traversal(ABC):
        def bypass(self, i1, i2, list, cfg, commitPath) -> int:
            raise NotImplementedError()

        def visit(self, cPath, commitToReport):
            cPath.append(commitToReport)

        def prepBypass(self, i1, i2, list, cfg):
            skipInterval = cfg["noCleanInterval"]
            cfg["serviceConfig"]["skipCleanInterval"] = i2 - i1 < skipInterval
            self.mode.commonLogger.info(
                "Check interval {i1}..{i2}".format(i1=i1, i2=i2)
            )
            self.mode.commonLogger.info(
                "Check commits {c1}..{c2}".format(c1=list[i1], c2=list[i2])
            )

        def __init__(self, mode) -> None:
            self.mode = mode

    class FirstFailedVersion(Traversal):
        def __init__(self, mode) -> None:
            super().__init__(mode)

        def bypass(self, i1, i2, list, cfg, commitPath) -> int:
            self.prepBypass(i1, i2, list, cfg)
            sampleCommit = 0
            if "sampleCommit" in cfg["serviceConfig"]:
                sampleCommit = cfg["serviceConfig"]["sampleCommit"]
            if i1 + 1 >= i2:
                isBad = self.mode.checkIfBordersDiffer(
                    sampleCommit, i1, list, cfg)
                breakCommit = i1 if isBad else i2
                pc = Mode.CommitPath.PathCommit(
                    breakCommit,
                    Mode.CommitPath.CommitState.BREAK
                )
                self.mode.setOutputInfo(pc)
                commitPath.accept(self, pc)
                return
            mid = (int)((i1 + i2) / 2)
            isBad = self.mode.checkIfBordersDiffer(
                    sampleCommit, mid, list, cfg)
            if isBad:
                self.bypass(
                    i1, mid, list, cfg, commitPath
                )
            else:
                self.bypass(
                    mid, i2, list, cfg, commitPath
                )

    class FirstFixedVersion(Traversal):
        def __init__(self, mode) -> None:
            super().__init__(mode)

        def bypass(self, i1, i2, list, cfg, commitPath) -> int:
            self.prepBypass(i1, i2, list, cfg)
            sampleCommit = 0
            if "sampleCommit" in cfg["serviceConfig"]:
                sampleCommit = cfg["serviceConfig"]["sampleCommit"]
            if i1 + 1 >= i2:
                isBad = self.mode.checkIfBordersDiffer(
                    sampleCommit, i1, list, cfg)
                breakCommit = i2 if isBad else i1
                pc = Mode.CommitPath.PathCommit(
                    breakCommit,
                    Mode.CommitPath.CommitState.BREAK
                )
                self.mode.setOutputInfo(pc)
                commitPath.accept(self, pc)
                return
            mid = (int)((i1 + i2) / 2)
            isBad = self.mode.checkIfBordersDiffer(
                    sampleCommit, mid, list, cfg)
            if isBad:
                self.bypass(
                    mid, i2, list, cfg, commitPath
                )
            else:
                self.bypass(
                    i1, mid, list, cfg, commitPath
                )

    class AllBreakVersions(Traversal):
        def __init__(self, mode) -> None:
            super().__init__(mode)

        def bypass(self, i1, i2, list, cfg, commitPath) -> int:
            self.prepBypass(i1, i2, list, cfg)
            sampleCommit = 0
            if "sampleCommit" in cfg["serviceConfig"]:
                sampleCommit = cfg["serviceConfig"]["sampleCommit"]
            if i1 + 1 >= i2:
                isBad = self.mode.checkIfBordersDiffer(
                    sampleCommit, i1, list, cfg)
                breakCommit = i1 if isBad else i2
                pc = Mode.CommitPath.PathCommit(
                    breakCommit,
                    Mode.CommitPath.CommitState.BREAK
                )
                self.mode.setOutputInfo(pc)
                commitPath.accept(self, pc)
                lastCommit = len(list) - 1
                isTailDiffer = self.mode.checkIfBordersDiffer(
                    breakCommit, lastCommit, list, cfg)
                if isTailDiffer:
                    cfg["serviceConfig"]["sampleCommit"] = breakCommit
                    self.bypass(
                       breakCommit, lastCommit,
                       list, cfg, commitPath
                    )
                return
            mid = (int)((i1 + i2) / 2)
            isBad = self.mode.checkIfBordersDiffer(
                    sampleCommit, mid, list, cfg)
            if isBad:
                self.bypass(
                    i1, mid, list, cfg, commitPath
                )
            else:
                self.bypass(
                    mid, i2, list, cfg, commitPath
                )
