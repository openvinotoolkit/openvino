import importlib
import os
import sys
import subprocess
import re
import json
import logging as log
from argparse import ArgumentParser

def getParams():
    parser = ArgumentParser()
    parser.add_argument("-c", "--commits", dest="commitSeq", help="commit sequence")
    parser.add_argument("-cfg", "--config", dest="configuration",
        help="configuration source", default="custom_cfg.json")
    parser.add_argument("-wd", "--workdir", dest="isWorkingDir",
        action='store_true', help="flag if current directory is working")
    args = parser.parse_args()

    presetCfgPath = "utils/cfg.json"
    customCfgPath = ""
    customCfgPath = args.__dict__["configuration"]
    cfgFile = open(presetCfgPath)
    presetCfgData = json.load(cfgFile)
    cfgFile.close()
    cfgFile = open(customCfgPath)
    customCfgData = json.load(cfgFile)
    cfgFile.close()
    # customize cfg
    for key in customCfgData:
        newVal = customCfgData[key]
        presetCfgData[key] = newVal

    presetCfgData = absolutizePaths(presetCfgData)
    return args, presetCfgData, customCfgPath

def absolutizePaths(cfg):
    pathToAbsolutize = ["gitPath", "buildPath", "appPath", "workPath"]
    for item in pathToAbsolutize:
        path = cfg[item]
        path = os.path.abspath(path)
        cfg[item] = path
    if "preprocess" in cfg["runConfig"]:
        prepFile = cfg["runConfig"]["preprocess"]["file"]
        prepFile = os.path.abspath(prepFile)
        cfg["runConfig"]["preprocess"]["file"] = prepFile
    return cfg
def checkArgAndGetCommitList(commitArg, cfgData):
    if (not len(commitArg.split('..')) == 2):
        # todo: python bug with re.search("^[a-zA-Z0-9]+\.\.[a-zA-Z0-9]+$", commitArg)
        raise ValueError("{arg} is not correct commit set".format(arg=commitArg))
    else:
        checkCommitSetCmd = "git log {commitInterval} --boundary --pretty=\"%h\"".format(commitInterval=commitArg)
        proc = subprocess.Popen(checkCommitSetCmd.split(),
            cwd = cfgData["gitPath"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        proc.wait()
        out, err = proc.communicate()
        out = out.decode('utf-8')
        outList = out.split()
        if (re.search(".*fatal.*", out)):
            print (out)
            raise ValueError("{arg} commit set is not valid".format(arg=commitArg))
        elif (len(outList) == 0):
            raise ValueError("{arg} commit set is empty".format(arg=commitArg))
        else:
            return outList
def runCommandList(commit, cfgData):
    skipCleanInterval = cfgData["trySkipClean"]
    commitLogger = getCommitLogger(cfgData, commit)
    commandList = cfgData["commandList"]
    gitPath = cfgData["gitPath"]
    buildPath = cfgData["buildPath"]
    defRepo = gitPath
    newEnv = os.environ.copy()
    for cmd in commandList:
        if "tag" in cmd:
            if cmd["tag"] == "clean" and skipCleanInterval:
                continue
            elif cmd["tag"] == "preprocess":
                if not ("preprocess" in cfgData["runConfig"] and
                    "name" in cfgData["runConfig"]["preprocess"]):
                    raise CfgError("No preprocess provided")
                prePrName = cfgData["runConfig"]["preprocess"]["name"]
                mod = importlib.import_module("utils.preprocess.{pp}".format(pp=prePrName))
                preProcess = getattr(mod, prePrName)
                preProcess(cfgData)
                continue
            elif cmd["tag"] == "setupenv":
                for env in cfgData["runConfig"]["setupenv"]:
                    envKey = env["env"]
                    envVal = env["val"]
                    commitLogger.info("Setup env: {key}={val}".format(key=envKey, val=envVal))
                    newEnv[envKey] = envVal
                continue
        makeCmd = cfgData["makeCmd"]
        strCommand = cmd["cmd"].format(commit=commit, makeCmd=makeCmd)
        formattedCmd = strCommand.split()
        cwd = defRepo
        if "path" in cmd:
            cwd = cmd["path"].format(buildPath = buildPath, gitPath = gitPath)
        commitLogger.info("Run command: {command}".format(command=formattedCmd))
        proc = subprocess.Popen(formattedCmd,
            cwd = cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=newEnv)
        for line in proc.stdout:
            # decode if line is byte-type
            try:
                line = line.decode('utf-8')
            except (UnicodeDecodeError, AttributeError):
                pass
            sys.stdout.write(line)
            commitLogger.info(line)
        proc.wait()
        checkOut, err = proc.communicate()
        try:
            checkOut = checkOut.decode('utf-8')
        except (UnicodeDecodeError, AttributeError):
            pass
        if "catchMsg" in cmd:
            isErrFound = re.search(cmd["catchMsg"], checkOut)
            if (isErrFound):
                if skipCleanInterval:
                    commitLogger.info("Build error: clean is necessary")
                    raise NoCleanFailedError()
                else:
                    raise CmdError(checkOut)

def fetchAppOutput(cfg):
    appCmd = cfg["appCmd"]
    appPath = cfg["appPath"]
    p = subprocess.Popen(appCmd.split(), cwd=appPath, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output, err = p.communicate()
    output = output.decode('utf-8')
    return output

def handleCommit(commit, cfgData):
    skipCleanInterval = cfgData["serviceConfig"]["skipCleanInterval"]
    cfgData["trySkipClean"] = skipCleanInterval
    try:
        runCommandList(commit, cfgData)
    except(NoCleanFailedError):
        cfgData["trySkipClean"] = False
        runCommandList(commit, cfgData)
def returnToActualVersion(cfg):
    cmd = cfg["returnCmd"]
    cwd = cfg["gitPath"]
    proc = subprocess.Popen(cmd.split(), cwd = cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    proc.wait()
    return
def setupLogger(name, logPath, logFileName, level=log.INFO):
    if not os.path.exists(logPath):
        os.makedirs(logPath)
    logFileName = logPath + logFileName
    open(logFileName, "w").close() # clear old log
    handler = log.FileHandler(logFileName)
    formatter = log.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger = log.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
def getCommitLogger(cfg, commit):
    logName = 'commitLogger_{c}'.format(c=commit)
    if log.getLogger(logName).hasHandlers():
        return log.getLogger(logName)
    logPath = getActualPath("logPath", cfg)
    logFileName = 'commit_{c}.log'.format(c=commit)
    commitLogger = setupLogger(logName, logPath, logFileName)
    return commitLogger
def getActualPath(pathName, cfg):
    workPath = cfg["workPath"]
    curPath = cfg[pathName]
    return curPath.format(workPath=workPath)
def safeClearDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    p = subprocess.Popen("rm -rf *", cwd=path, stdout=subprocess.PIPE, shell=True)
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
def checkAndGetClassnameByConfig(cfg, mapName, specialCfg):
    keyName = cfg["runConfig"][specialCfg]
    map = cfg[mapName]
    if (not (keyName in map)):
        raise CfgError("{keyName} is not registered in {mapName}".format(keyName = keyName, mapName = mapName))
    else:
        return map[keyName]
def checkAndGetSubclass(className, parentClass):
    cl = [cl for cl in parentClass.__subclasses__() if cl.__name__ == className]
    if not(cl.__len__() == 1):
        raise CfgError("Class {className} doesn't exist".format(className = className))
    else:
        return cl[0]
