import importlib
import shutil
import os
import sys
import subprocess
import string
from enum import Enum
import re
import json
import logging as log
from argparse import ArgumentParser


def getMeaningfullCommitTail(commit):
    return commit[:7]


def getParams():
    parser = ArgumentParser()
    parser.add_argument("-c", "--commits", dest="commitSeq", help="commit set")
    parser.add_argument(
        "-cfg",
        "--config",
        dest="configuration",
        help="configuration source",
        default="custom_cfg.json",
    )
    parser.add_argument(
        "-wd",
        "--workdir",
        dest="isWorkingDir",
        action="store_true",
        help="flag if current directory is working",
    )
    args = parser.parse_args()

    presetCfgPath = "utils/cfg.json"
    customCfgPath = ""
    customCfgPath = args.__dict__["configuration"]
    presetCfgData = None
    with open(presetCfgPath) as cfgFile:
        presetCfgData = json.load(cfgFile)
    cfgFile.close()
    customCfgData = None
    with open(customCfgPath) as cfgFile:
        customCfgData = json.load(cfgFile)
    cfgFile.close()
    # customize cfg
    for key in customCfgData:
        newVal = customCfgData[key]
        presetCfgData[key] = newVal

    presetCfgData = absolutizePaths(presetCfgData)
    return args, presetCfgData, customCfgPath


def getBlobDiff(file1, file2):
    with open(file1) as file:
        content = file.readlines()
    with open(file2) as sampleFile:
        sampleContent = sampleFile.readlines()
    # ignore first line with memory address
    i = -1
    curMaxDiff = 0
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
            curMaxDiff = max(curMaxDiff, abs(val - sampleVal))
    return curMaxDiff


def absolutizePaths(cfg):
    pl = sys.platform
    if pl == "linux" or pl == "linux2":
        cfg["workPath"] = cfg["linWorkPath"]
        cfg["os"] = "linux"
    elif pl == "win32":
        wp = cfg["winWorkPath"]
        wp = "echo {path}".format(path=wp)
        wp = subprocess.check_output(wp, shell=True)
        wp = wp.decode()
        wp = wp.rstrip()
        cfg["workPath"] = wp
        cfg["os"] = "win"
    else:
        raise CfgError(
            "No support for current OS: {pl}".format(pl=pl)
            )
    if cfg["dlbConfig"]["launchedAsJob"]:
        cfg["appPath"] = cfg["dlbConfig"]["appPath"]
    pathToAbsolutize = ["gitPath", "buildPath", "appPath", "workPath"]
    for item in pathToAbsolutize:
        path = cfg[item]
        path = os.path.abspath(path)
        cfg[item] = path
    if "preprocess" in cfg["runConfig"] and "file" in cfg["runConfig"]["preprocess"]:
        prepFile = cfg["runConfig"]["preprocess"]["file"]
        prepFile = os.path.abspath(prepFile)
        cfg["runConfig"]["preprocess"]["file"] = prepFile
    if "envVars" in cfg:
        updatedEnvVars = []
        for env in cfg["envVars"]:
            envKey = env["name"]
            envVal = env["val"]
            # format ov-path in envvars for e2e case
            if "{gitPath}" in envVal:
                envVal = envVal.format(gitPath=cfg["gitPath"])
                envVal = os.path.abspath(envVal)
                updatedVar = {"name": envKey, "val": envVal}
                updatedEnvVars.append(updatedVar)
            else:
                updatedEnvVars.append(env)
        cfg["envVars"] = updatedEnvVars
    return cfg


def checkArgAndGetCommits(commArg, cfgData):
    # WA because of python bug with
    # re.search("^[a-zA-Z0-9]+\.\.[a-zA-Z0-9]+$", commArg)
    if not len(commArg.split("..")) == 2:
        raise ValueError("{arg} is not correct commit set".format(arg=commArg))
    else:
        getCommitSetCmd = 'git log {interval} --boundary --pretty="%h"'.format(
            interval=commArg
        )
        proc = subprocess.Popen(
            getCommitSetCmd.split(),
            cwd=cfgData["gitPath"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        proc.wait()
        out, err = proc.communicate()
        out = out.decode("utf-8")
        outList = out.split()
        if re.search(".*fatal.*", out):
            print(out)
            raise ValueError("{arg} commit set is invalid".format(arg=commArg))
        elif len(outList) == 0:
            raise ValueError("{arg} commit set is empty".format(arg=commArg))
        else:
            return outList


def runCommandList(commit, cfgData, enforceClean=False):
    skipCleanInterval = False
    if "trySkipClean" not in cfgData:
        skipCleanInterval = not enforceClean
    else:
        skipCleanInterval = cfgData["trySkipClean"] and not enforceClean
    commitLogger = getCommitLogger(cfgData, commit)
    if not cfgData["extendBuildCommand"]:
        commandList = cfgData["commandList"]
    else:
        commandList = cfgData["extendedCommandList"]
    gitPath = cfgData["gitPath"]
    buildPath = cfgData["buildPath"]
    defRepo = gitPath
    for cmd in commandList:
        if "tag" in cmd:
            if cmd["tag"] == "preprocess":
                if not (
                    "preprocess" in cfgData["runConfig"]
                    and "name" in cfgData["runConfig"]["preprocess"]
                ):
                    raise CfgError("No preprocess provided")
                prePrName = cfgData["runConfig"]["preprocess"]["name"]
                mod = importlib.import_module(
                    "utils.preprocess.{pp}".format(pp=prePrName)
                )
                preProcess = getattr(mod, prePrName)
                preProcess(cfgData, commit)
                continue
        makeCmd = cfgData["makeCmd"]
        # {commit}, {makeCmd}, {cashedPath} placeholders
        strCommand = cmd["cmd"].format(commit=commit, makeCmd=makeCmd)
        pathExists, cashedPath = getCashedPath(commit, cfgData)
        if pathExists:
            strCommand = cmd["cmd"].format(cashedPath=cashedPath)
        formattedCmd = strCommand.split()
        # define command launch destination
        cwd = defRepo
        if "path" in cmd:
            cwd = cmd["path"].format(buildPath=buildPath, gitPath=gitPath)
        # run and check
        commitLogger.info("Run command: {command}".format(
            command=formattedCmd)
        )
        proc = subprocess.Popen(
            formattedCmd, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace"
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            commitLogger.info(line)
            if "catchMsg" in cmd:
                isErrFound = re.search(cmd["catchMsg"], line)
                if isErrFound:
                    raise BuildError(
                        errType=BuildError.BuildErrType.UNDEFINED,
                        message="error while executing: {}".
                            format(cmd["cmd"]), commit=commit
                        )
        proc.wait()
        checkOut, err = proc.communicate()


def fetchAppOutput(cfg, commit):
    commitLogger = getCommitLogger(cfg, commit)
    appPath = cfg["appPath"]
    # format appPath if it was cashed
    if cfg["cachedPathConfig"]["enable"] == True:
        pathExists, suggestedAppPath = getCashedPath(commit, cfg)
        if pathExists:
            for item in string.Formatter().parse(appPath):
                if item[1] is not None and item[1] == 'cashedPath':
                    commitLogger.info(
                        "App path, corresponding commit {c} is cashed, "
                        "value:{p}".format(c=commit,
                                           p=suggestedAppPath))
            appPath.format(cashedPath=suggestedAppPath)
    newEnv = os.environ.copy()
    if "envVars" in cfg:
        for env in cfg["envVars"]:
            envKey = env["name"]
            envVal = env["val"]
            newEnv[envKey] = envVal
    appCmd = cfg["appCmd"]
    commitLogger.info("Run command: {command}".format(
        command=appCmd)
    )
    shellFlag = True
    if cfg["os"] == "linux":
        shellFlag = False
    p = subprocess.Popen(
        appCmd.split(),
        cwd=appPath,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=newEnv,
        shell=shellFlag
    )
    output, err = p.communicate()
    output = output.decode("utf-8")
    return output


def handleCommit(commit, cfgData):
    commitLogger = getCommitLogger(cfgData, commit)
    cashedPath = None
    if cfgData["cachedPathConfig"]["enable"] == True:
        pathExists, cashedPath = getCashedPath(commit, cfgData)
        if pathExists:
            commitLogger.info(
                "Path, corresponding commit {c} is cashed, value:{p}".format(
                c=commit, p=cashedPath))
            return
        else:
            if cfgData["cachedPathConfig"]["scheme"] == "mandatory":
                commitLogger.info("Ignore commit {}".format(commit))
                raise BuildError(
                    errType=BuildError.BuildErrType.TO_IGNORE,
                    message="build error handled by skip",
                    commit=commit
                    )
            else:
                raise BuildError(
                    errType=BuildError.BuildErrType.UNSUPPORTED,
                    message="optional scheme of cashedAppPath is to-be implemented",
                    commit=commit
                    )

    if "skipCleanInterval" in cfgData["serviceConfig"]:
        skipCleanInterval = cfgData["serviceConfig"]["skipCleanInterval"]
        cfgData["trySkipClean"] = skipCleanInterval
    try:
        runCommandList(commit, cfgData)
        if cfgData["skipMode"]["flagSet"]["enableRebuild"]:
            cfgData["skipMode"]["flagSet"]["switchOnSimpleBuild"] = True
            cfgData["skipMode"]["flagSet"]["switchOnExtendedBuild"] = False
            cfgData["extendBuildCommand"] = False
    except BuildError as be:
        if cfgData["skipMode"]["flagSet"]["enableSkips"]:
            commitLogger.info("Build error: commit {} skipped".format(commit))
            raise BuildError(
                errType=BuildError.BuildErrType.TO_SKIP,
                message="build error handled by skip",
                commit=commit
                ) from be
        elif cfgData["skipMode"]["flagSet"]["enableRebuild"]:
            if cfgData["skipMode"]["flagSet"]["switchOnSimpleBuild"]:
                cfgData["skipMode"]["flagSet"]["switchOnSimpleBuild"] = False
                cfgData["skipMode"]["flagSet"]["switchOnExtendedBuild"] = True
                commitLogger.info("Build error: commit {} rebuilded".format(commit))
                raise BuildError(
                    errType=BuildError.BuildErrType.TO_REBUILD,
                    message="build error handled by rebuilding",
                    commit=commit
                    ) from be
            elif cfgData["skipMode"]["flagSet"]["switchOnExtendedBuild"]:
                raise BuildError(
                    errType=BuildError.BuildErrType.TO_STOP,
                    message="cannot rebuild commit",
                    commit=commit
                    ) from be
            else:
                raise BuildError(
                    errType=BuildError.BuildErrType.WRONG_STATE,
                    message="incorrect case with commit",
                    commit=commit
                    ) from be
        else:
            raise BuildError(
                        message = "error occured during handling",
                        errType = BuildError.BuildErrType.WRONG_STATE,
                        commit=commit
                        )


def getCashedPath(commit, cfgData):
    shortHash = getMeaningfullCommitTail(commit)
    if commit in cfgData["cachedPathConfig"]["cashMap"]:
        return True, cfgData["cachedPathConfig"]["cashMap"][commit]
    else:
        return False, None


def returnToActualVersion(cfg):
    cmd = cfg["returnCmd"]
    cwd = cfg["gitPath"]
    proc = subprocess.Popen(
        cmd.split(), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    proc.wait()
    return


def setupLogger(name, logPath, logFileName, level=log.INFO):
    if not os.path.exists(logPath):
        os.makedirs(logPath)
    logFileName = logPath + logFileName
    with open(logFileName, "w"):  # clear old log
        pass
    handler = log.FileHandler(logFileName)
    formatter = log.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger = log.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def getCommitLogger(cfg, commit):
    logName = "commitLogger_{c}".format(c=commit)
    if log.getLogger(logName).hasHandlers():
        return log.getLogger(logName)
    logPath = getActualPath("logPath", cfg)
    logFileName = "commit_{c}.log".format(c=commit)
    commitLogger = setupLogger(logName, logPath, logFileName)
    return commitLogger


def getActualPath(pathName, cfg):
    workPath = cfg["workPath"]
    curPath = cfg[pathName]
    return curPath.format(workPath=workPath)


def safeClearDir(path, cfg):
    if not os.path.exists(path):
        os.makedirs(path)
    if cfg["os"] == "win":
        shutil.rmtree(path)
    else:
        # WA, because of unstability of rmtree()
        # in linux environment
        p = subprocess.Popen(
            "rm -rf *", cwd=path,
            stdout=subprocess.PIPE, shell=True
        )
        p.wait()
    return


class CfgError(Exception):
    pass


class CashError(Exception):
    pass


class CmdError(Exception):
    pass


class RepoError(Exception):
    pass


class BuildError(Exception):
    class BuildErrType(Enum):
        # Undefined - unresolved behaviour, to-do ...
        UNDEFINED = 0
        # strategies to handle unsuccessful build
        TO_REBUILD = 1
        TO_SKIP = 2
        TO_STOP = 3
        # commit supposed to contain irrelevant change,
        # build is unnecessary
        TO_IGNORE = 4
        # throwed in unexpected case
        WRONG_STATE = 5
        # state handling unsupported, i.e., 'optional'
        # scheme of cashedAppPath handling is to-be implemented
        UNSUPPORTED = 6
    def __init__(self, commit, message, errType):
        self.message = message
        self.errType = errType
        self.commit = commit
    def __str__(self):
        return self.message


def checkAndGetClassnameByConfig(cfg, mapName, specialCfg):
    keyName = cfg["runConfig"][specialCfg]
    map = cfg[mapName]
    if not (keyName in map):
        raise CfgError(
            "{keyName} is not registered in {mapName}".format(
                keyName=keyName, mapName=mapName
            )
        )
    else:
        return map[keyName]


def checkAndGetSubclass(clName, parentClass):
    cl = [cl for cl in parentClass.__subclasses__() if cl.__name__ == clName]
    if not (cl.__len__() == 1):
        raise CfgError("Class {clName} doesn't exist".format(clName=clName))
    else:
        return cl[0]
