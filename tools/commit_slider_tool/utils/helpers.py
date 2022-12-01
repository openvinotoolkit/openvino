import importlib
import os
import sys
import subprocess
import re
import logging as log

def checkArgAndGetCommitList(commitArg, cfgData):
    if (not len(commitArg.split('..')) == 2):
        # todo: python bug with re.search("^[a-zA-Z0-9]+\.\.[a-zA-Z0-9]+$", commitArg)
        raise ValueError("{arg} is not corect commit set".format(arg=commitArg))
    else:
        checkCommitSetCmd = "git log {commitInterval} --boundary --pretty=\"%h\"".format(commitInterval=commitArg)
        proc = subprocess.Popen(checkCommitSetCmd.split(),
            cwd = cfgData["commonConfig"]["gitPath"],
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
    commandList = cfgData["commonConfig"]["commandList"]
    gitPath = cfgData["commonConfig"]["gitPath"]
    buildPath = cfgData["commonConfig"]["buildPath"]
    defRepo = gitPath
    for cmd in commandList:
        if "tag" in cmd.keys():
            if cmd["tag"] == "clean" and skipCleanInterval:
                continue
            elif cmd["tag"] == "preprocess":
                if not ("preprocess" in cfgData["specialConfig"].keys() and 
                    "name" in cfgData["specialConfig"]["preprocess"].keys()):
                    raise CfgError("No preprocess provided")
                prePrName = cfgData["specialConfig"]["preprocess"]["name"]
                mod = importlib.import_module("utils.preprocess.{pp}".format(pp=prePrName))
                preProcess = getattr(mod, prePrName)
                preProcess(cfgData)
                continue
        strCommand = cmd["cmd"].format(commit = commit)
        formattedCmd = strCommand.split()
        cwd = defRepo
        if "path" in cmd.keys():
            cwd = cmd["path"].format(buildPath = buildPath, gitPath = gitPath)
        commitLogger.info("Run command: {command}".format(command=formattedCmd))
        proc = subprocess.Popen(formattedCmd,
            cwd = cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            # decode if line is byte-type
            try:
                line = line.decode('utf-8')
            except (UnicodeDecodeError, AttributeError):
                pass
            sys.stdout.write(line)
        proc.wait()
        i = input("wait for you")
        checkOut, err = proc.communicate()
        commitLogger.info(checkOut)
        if "catchMsg" in cmd.keys():
            isErrFound = re.search(cmd["catchMsg"], checkOut.decode('utf-8'))
            if (isErrFound):
                if skipCleanInterval:
                    commitLogger.info("Build error: clean is necessary")
                    raise NoCleanFailedError()
                else:
                    raise CmdError(checkOut)

def handleCommit(commit, cfgData):
    skipCleanInterval = cfgData["serviceConfig"]["skipCleanInterval"]
    cfgData["trySkipClean"] = skipCleanInterval
    try:
        runCommandList(commit, cfgData)
    except(NoCleanFailedError):
        cfgData["trySkipClean"] = False
        runCommandList(commit, cfgData)
def returnToActualVersion(cfg):
    cmd = cfg["commonConfig"]["returnCmd"]
    cwd = cfg["commonConfig"]["gitPath"]
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
    workPath = cfg["commonConfig"]["workPath"]
    curPath = cfg["commonConfig"][pathName]
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
    keyName = cfg["specialConfig"][specialCfg]
    map = cfg["commonConfig"][mapName]
    if (not (keyName in map.keys())):
        raise CfgError("{keyName} is not registered in {mapName}".format(keyName = keyName, mapName = mapName))
    else:
        return map[keyName]
def checkAndGetSubclass(className, parentClass):
    cl = [cl for cl in parentClass.__subclasses__() if cl.__name__ == className]
    if not(cl.__len__() == 1):
        raise CfgError("Class {className} doesn't exist".format(className = className))
    else:
        return cl[0]