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

    def getPseudoMetric(self, commit, cfg):
        raise NotImplementedError("getPseudoMetric() is not implemented")

    def compareCommits(self, c1, c2, list, cfg):
        raise NotImplementedError("compareCommits() is not implemented")

    def checkCfg(self, cfg):
        if not ("traversal" in cfg["runConfig"]):
            raise util.CfgError("traversal is not configured")

    def prepareRun(self, list, cfg):
        self.normalizeCfg(cfg)
        cfg["serviceConfig"] = {}
        if cfg["checkIfBordersDiffer"] and not self.checkIfListBordersDiffer(
                list, cfg):
            raise util.RepoError("Borders {i1} and {i2} doesn't differ".format(
                i1=0, i2=len(list) - 1))
        self.commitList = list

    def normalizeCfg(self, cfg):
        if not self.traversal.isComparative():
            cfg["checkIfBordersDiffer"] = False
        if "modeName" in cfg["skipMode"]:
            errorHandlingMode = cfg["skipMode"]["modeName"]
            if errorHandlingMode == "skip":
                cfg["skipMode"]["flagSet"] = {}
                cfg["skipMode"]["flagSet"]["enableSkips"] = True
            elif (errorHandlingMode == "rebuild"):
                cfg["skipMode"]["flagSet"] = {}
                cfg["skipMode"]["flagSet"]["enableSkips"] = False
                cfg["skipMode"]["flagSet"]["enableRebuild"] = True
                cfg["skipMode"]["flagSet"]["switchOnSimpleBuild"] = True
                cfg["skipMode"]["flagSet"]["switchOnExtendedBuild"] = False
            else:
                raise util.CfgError(
                    "Error handling mode {} is not supported".format(errorHandlingMode)
                    )

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

    def run(self, list, cfg) -> int:
        self.prepareRun(list, cfg)
        for i, item in enumerate(list):
            list[i] = item.replace('"', "")
        self.traversal.wrappedBypass(
            list, list, cfg
        )
        self.postRun(list)

    def setOutputInfo(self, pathCommit):
        # override if you need more details in output representation
        pass

    def printResult(self):
        for pathcommit in self.commitPath.getList():
            print(self.getCommitInfo(pathcommit))

    def getCommitInfo(self, commit):
        # override if you need more details in output representation
        return "Break commit: {c}, state: {s}".format(
            c=commit.cHash, s=commit.state)

    def checkIfListBordersDiffer(self, list, cfg):
        return self.compareCommits(list[0], list[-1], list, cfg)

    class CommitPath:

        def __init__(self):
            self.commitList = []

        def accept(self, traversal, commitToReport) -> None:
            traversal.visit(self, commitToReport)

        class CommitState(Enum):
            BREAK = 1
            SKIPPED = 2
            IGNORED = 3

        class PathCommit:
            def __init__(self, cHash, state):
                self.cHash = cHash
                self.state = state

        def append(self, commit):
            self.commitList.append(commit)

        def pop(self):
            return self.commitList.pop(0)

        def getList(self):
            return self.commitList

    class Traversal(ABC):
        def bypass(self, curList, list, cfg) -> int:
            raise NotImplementedError()

        def wrappedBypass(self, curList, list, cfg) -> int:
            try:
                self.bypass(curList, list, cfg)
            except util.BuildError as be:
                if be.errType == util.BuildError.BuildErrType.TO_SKIP:
                    self.skipCommit(be.commit, curList, cfg)
                    self.wrappedBypass(curList, list, cfg)
                if be.errType == util.BuildError.BuildErrType.TO_IGNORE:
                    self.ignoreCommit(be.commit, curList, cfg)
                    self.wrappedBypass(curList, list, cfg)
                elif be.errType == util.BuildError.BuildErrType.TO_REBUILD:
                    cfg["extendBuildCommand"] = True
                    self.wrappedBypass(curList, list, cfg)
                else:
                    # exception must be reported to user
                    pass


        def skipCommit(self, commit, curList, cfg):
            curList.remove(commit)
            pc = Mode.CommitPath.PathCommit(
                commit,
                Mode.CommitPath.CommitState.SKIPPED
            )
            self.mode.commonLogger.info(
                "Skipped commit {}".format(commit)
            )
            self.mode.setOutputInfo(pc)
            self.mode.commitPath.accept(self, pc)


        def ignoreCommit(self, commit, curList, cfg):
            curList.remove(commit)
            pc = Mode.CommitPath.PathCommit(
                commit,
                Mode.CommitPath.CommitState.IGNORED
            )
            self.mode.commonLogger.info(
                "Ignored commit {}".format(commit)
            )
            self.mode.setOutputInfo(pc)
            self.mode.commitPath.accept(self, pc)


        def visit(self, cPath, commitToReport):
            cPath.append(commitToReport)


        def prepBypass(self, curList, list, cfg):
            skipInterval = cfg["noCleanInterval"]
            i1 = list.index(curList[0])
            i2 = list.index(curList[-1])
            cfg["serviceConfig"]["skipCleanInterval"] = i2 - i1 < skipInterval
            self.mode.commonLogger.info(
                "Check interval {i1}..{i2}".format(i1=i1, i2=i2)
            )
            self.mode.commonLogger.info(
                "Check commits {c1}..{c2}".format(c1=list[i1], c2=list[i2])
            )

        def isComparative(self):
            # redefine for uncommon traversal
            return True


        def __init__(self, mode) -> None:
            self.mode = mode

    class FirstFailedVersion(Traversal):
        def __init__(self, mode) -> None:
            super().__init__(mode)

        def bypass(self, curList, list, cfg) -> int:
            self.prepBypass(curList, list, cfg)
            sampleCommit = curList[0]
            curLen = len(curList)
            if "sampleCommit" in cfg["serviceConfig"]:
                sampleCommit = cfg["serviceConfig"]["sampleCommit"]
            if curLen <= 2:
                isBad = self.mode.compareCommits(
                    sampleCommit, curList[0], list, cfg)
                breakCommit = curList[0] if isBad else curList[-1]
                pc = Mode.CommitPath.PathCommit(
                    breakCommit,
                    Mode.CommitPath.CommitState.BREAK
                )
                self.mode.setOutputInfo(pc)
                self.mode.commitPath.accept(self, pc)
                return
            mid = (int)((curLen - 1) / 2)
            isBad = self.mode.compareCommits(
                    sampleCommit, curList[mid], list, cfg)
            if isBad:
                self.wrappedBypass(
                    curList[0 : mid + 1], list, cfg
                )
            else:
                self.wrappedBypass(
                    curList[mid :], list, cfg
                )

    class FirstFixedVersion(Traversal):
        def __init__(self, mode) -> None:
            super().__init__(mode)

        def bypass(self, curList, list, cfg) -> int:
            self.prepBypass(curList, list, cfg)
            sampleCommit = curList[0]
            curLen = len(curList)
            if "sampleCommit" in cfg["serviceConfig"]:
                sampleCommit = cfg["serviceConfig"]["sampleCommit"]
            if curLen <= 2:
                isBad = self.mode.compareCommits(
                    sampleCommit, curList[0], list, cfg)
                breakCommit = curList[-1] if isBad else curList[0]
                pc = Mode.CommitPath.PathCommit(
                    breakCommit,
                    Mode.CommitPath.CommitState.BREAK
                )
                self.mode.setOutputInfo(pc)
                self.mode.commitPath.accept(self, pc)
                return
            mid = (int)((curLen - 1) / 2)
            isBad = self.mode.compareCommits(
                    sampleCommit, curList[mid], list, cfg)
            if isBad:
                self.wrappedBypass(
                    curList[mid :], list, cfg
                )
            else:
                self.wrappedBypass(
                    curList[0 : mid + 1], list, cfg
                )

    class AllBreakVersions(Traversal):
        def __init__(self, mode) -> None:
            super().__init__(mode)

        def bypass(self, curList, list, cfg) -> int:
            self.prepBypass(curList, list, cfg)
            sampleCommit = curList[0]
            curLen = len(curList)
            if "sampleCommit" in cfg["serviceConfig"]:
                sampleCommit = cfg["serviceConfig"]["sampleCommit"]
            if curLen <= 2:
                isBad = self.mode.compareCommits(
                    sampleCommit, curList[0], list, cfg)
                breakCommit = curList[0] if isBad else curList[-1]
                pc = Mode.CommitPath.PathCommit(
                    breakCommit,
                    Mode.CommitPath.CommitState.BREAK
                )
                self.mode.setOutputInfo(pc)
                self.mode.commitPath.accept(self, pc)
                lastCommit = list[-1]
                isTailDiffer = self.mode.compareCommits(
                    breakCommit, lastCommit, list, cfg)
                if isTailDiffer:
                    cfg["serviceConfig"]["sampleCommit"] = breakCommit
                    # to-do make copy without skip-commits
                    brIndex = list.index(breakCommit)
                    self.wrappedBypass(
                       list[brIndex :],
                       list, cfg
                    )
                return
            mid = (int)((curLen - 1) / 2)
            isBad = self.mode.compareCommits(
                    sampleCommit, curList[mid], list, cfg)
            if isBad:
                self.wrappedBypass(
                    curList[0 : mid + 1], list, cfg
                )
            else:
                self.wrappedBypass(
                    curList[mid :], list, cfg
                )

    class BruteForce(Traversal):
        def __init__(self, mode) -> None:
            super().__init__(mode)

        def bypass(self, curList, list, cfg) -> int:
            for commit in list:
                self.mode.commonLogger.info(
                    "Handle commit {}".format(commit))
                self.mode.getPseudoMetric(commit, cfg)

        def isComparative(self):
            return False
