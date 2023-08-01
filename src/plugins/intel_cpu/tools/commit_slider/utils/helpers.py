import importlib
import shutil
import os
import sys
import subprocess
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
    commandList = cfgData["commandList"]
    gitPath = cfgData["gitPath"]
    buildPath = cfgData["buildPath"]
    defRepo = gitPath
    for cmd in commandList:
        if "tag" in cmd:
            if cmd["tag"] == "clean" and skipCleanInterval:
                continue
            elif cmd["tag"] == "preprocess":
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
        strCommand = cmd["cmd"].format(commit=commit, makeCmd=makeCmd)
        formattedCmd = strCommand.split()
        cwd = defRepo
        if "path" in cmd:
            cwd = cmd["path"].format(buildPath=buildPath, gitPath=gitPath)
        commitLogger.info("Run command: {command}".format(
            command=formattedCmd)
        )
        proc = subprocess.Popen(
            formattedCmd, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        for line in proc.stdout:
            # decode if line is byte-type
            try:
                line = line.decode("utf-8")
            except (UnicodeDecodeError, AttributeError):
                pass
            sys.stdout.write(line)
            commitLogger.info(line)
        proc.wait()
        checkOut, err = proc.communicate()
        try:
            checkOut = checkOut.decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            pass
        if "catchMsg" in cmd:
            isErrFound = re.search(cmd["catchMsg"], checkOut)
            if isErrFound:
                if skipCleanInterval:
                    commitLogger.info("Build error: clean is necessary")
                    raise NoCleanFailedError()
                else:
                    raise CmdError(checkOut)


def fetchAppOutput(cfg, commit):
    newEnv = os.environ.copy()
    if "envVars" in cfg:
        for env in cfg["envVars"]:
            envKey = env["name"]
            envVal = env["val"]
            newEnv[envKey] = envVal
    appCmd = cfg["appCmd"]
    appPath = cfg["appPath"]
    commitLogger = getCommitLogger(cfg, commit)
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
    if "skipCleanInterval" in cfgData["serviceConfig"]:
        skipCleanInterval = cfgData["serviceConfig"]["skipCleanInterval"]
        cfgData["trySkipClean"] = skipCleanInterval
    try:
        runCommandList(commit, cfgData)
    except (NoCleanFailedError):
        cfgData["trySkipClean"] = False
        runCommandList(commit, cfgData)


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


class NoCleanFailedError(Exception):
    pass


class RepoError(Exception):
    pass


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
