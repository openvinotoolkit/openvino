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
def handleCommit(commit, cfgData):
    commitLogger = getCommitLogger(cfgData, commit)
    commandList = cfgData["commonConfig"]["commandList"]
    gitPath = cfgData["commonConfig"]["gitPath"]
    buildPath = cfgData["commonConfig"]["buildPath"]
    defRepo = gitPath
    for cmd in commandList:
        strCommand = cmd["cmd"].format(commit = commit)
        formattedCmd = strCommand.split()
        cwd = defRepo
        if "path" in cmd.keys():
            cwd = cmd["path"].format(buildPath = buildPath, gitPath = gitPath)
        proc = subprocess.Popen(formattedCmd,
            cwd = cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        proc.wait()
        checkOut, err = proc.communicate()
        commitLogger.info(checkOut)
        if "catchMsg" in cmd.keys():
            isErrFound = re.search(cmd["catchMsg"], checkOut.decode('utf-8'))
            if (isErrFound):
                raise CmdError(checkOut)
def setupLogger(name, logFile, level=log.INFO):
    open(logFile, "w").close() # clear old log
    handler = log.FileHandler(logFile)
    formatter = log.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger = log.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
def getCommitLogger(cfg, commit):
    logPath=cfg["commonConfig"]["logPath"]
    commitLogger = setupLogger(
        'commitLogger_{c}'.format(c=commit),
        '{logPath}/commit_{c}.log'.format(c=commit, logPath=logPath))
    return commitLogger
class CfgError(Exception):
    pass
class CashError(Exception):
    pass
class CmdError(Exception):
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