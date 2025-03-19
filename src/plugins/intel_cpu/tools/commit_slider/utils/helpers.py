# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import shutil
import os
import sys
import subprocess
from enum import Enum
import re
import json
import logging as log
from argparse import ArgumentParser
from utils.cfg_manager import CfgManager
import copy


mulKey = 'multiplication_key'

def getMeaningfullCommitTail(commit):
    return commit[:7]


def extractModelPath(cmdStr):
    args = cmdStr.split()
    return args[args.index("-m") + 1]


def getActualCfg(cfg, multiconfig: str):
    if isinstance(cfg, list) and \
        multiconfig != 'undefined':
        return cfg[int(multiconfig)]
    else:
        return cfg


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
    parser.add_argument(
        "-u",
        "--utility",
        dest="utility",
        help="run utility with specified name",
        default="no_utility",
    )

    parser.add_argument(
        "-x",
        "--multiconfig",
        dest="multiconfig",
        help="index in config array or 'undefined' mark",
        default="undefined"
    )
    args, additionalArgs = parser.parse_known_args()

    argHolder = DictHolder(args.__dict__)

    presetCfgPath = "utils/cfg.json"
    customCfgPath = ""
    customCfgPath = argHolder.configuration
    presetCfgData = loadJSONToString(presetCfgPath)

    if argHolder.utility != "no_utility":
        it = iter(additionalArgs)
        addDict = dict(zip(it, it))
        mergedArgs = {**(args.__dict__), **addDict}
        argHolder = DictHolder(mergedArgs)
        presetCfgData = loadJSONToObject(presetCfgPath)
        return argHolder, presetCfgData, presetCfgPath

    customCfgData = loadJSONToString(customCfgPath)
    if mulKey in customCfgData:
        customCfgData = multiplyCfgByKey(json.loads(customCfgData))
    else:
        customCfgData = json.loads(customCfgData)

    presetCfgData = customizeCfg(customCfgData, presetCfgData)

    return argHolder, presetCfgData, customCfgPath


def loadJSONToString(path):
    with open(path, 'r') as file:
        data = file.read()
    file.close()
    return data


def loadJSONToObject(path):
    with open(path) as file:
        obj = json.load(file)
    file.close()
    return obj


def customizeCfg(customCfg, presetCfg: str):
    if isinstance(customCfg, list):
        returnCfgList = [{}] * len(customCfg)
        for idx, subCfg in enumerate(customCfg):
            returnCfgList[idx] = customizeCfg(subCfg, presetCfg)
        return returnCfgList
    else:
        presetCfg = json.loads(presetCfg)

        # config manager resolves templates in config,
        # in the future, all interactions with config will
        # be incapsulated in config manager
        cm = CfgManager(customCfg)
        customCfg = cm.applyTemplate()

        # customize cfg
        for key in customCfg:
            newVal = customCfg[key]
            presetCfg[key] = newVal

        presetCfg = absolutizePaths(presetCfg)
        return presetCfg


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
    if pl in ["linux", "linux2", "darwin"]:
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
            # todo: move to substitute rules
            # format ov-path in envvars for e2e case
            if "{gitPath}" in envVal:
                envVal = envVal.format(gitPath=cfg["gitPath"])
                envVal = os.path.abspath(envVal)
                updatedVar = {"name": envKey, "val": envVal}
                updatedEnvVars.append(updatedVar)
            else:
                updatedEnvVars.append(env)
        cfg["envVars"] = updatedEnvVars
    cfg["venvCfg"]["venvDir"] = cfg["venvCfg"]["venvDir"].format(
        workPath=cfg["workPath"]
    )
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


def runCommandList(commit, cfgData):
    commitLogger = getCommitLogger(cfgData, commit)
    if not cfgData["extendBuildCommand"]:
        commandList = cfgData["commandList"]
    else:
        commandList = cfgData["extendedCommandList"]
    # apply substitution rules
    applySubstitutionRules(cfgData, cfgData["substitutionRules"], commit)
    commitLogger.info("Actual config is: {}".format(cfgData))

    gitPath = cfgData["gitPath"]
    buildPath = cfgData["buildPath"]
    defRepo = gitPath
    newEnv = os.environ.copy()
    if "buildEnvVars" in cfgData:
        for env in cfgData["buildEnvVars"]:
            envKey = env["name"]
            envVal = env["val"]
            newEnv[envKey] = envVal
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
        # todo - change to substitution rules
        pathExists, cashedPath = getCashedPath(commit, cfgData)
        if pathExists:
            # todo - and {} in cmd
            strCommand = cmd["cmd"].format(
                cashedPath=cashedPath,
                commit=commit, makeCmd=makeCmd)
        else:
            strCommand = cmd["cmd"].format(
                commit=commit, makeCmd=makeCmd)
        formattedCmd = strCommand.split()
        # define command launch destination
        cwd = defRepo
        if "path" in cmd:
            # todo - cashedpath
            cwd = cmd["path"].format(
                buildPath=buildPath,
                gitPath=gitPath,
                cashedPath=cashedPath)
        # run and check
        commitLogger.info("Run command: {command}".format(
            command=formattedCmd)
        )
        proc = subprocess.Popen(
            formattedCmd, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace",
            env=newEnv
        )
        for line in proc.stdout:
            if cfgData["verboseOutput"]:
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
    if cfg["cachedPathConfig"]["enabled"] == True:
        pathExists, suggestedAppPath = getCashedPath(commit, cfg)
        if pathExists and cfg["cachedPathConfig"]["changeAppPath"]:
            commitLogger.info(
                "App path, corresponding commit {c} is cashed, "
                "value:{p}".format(c=commit, p=suggestedAppPath))
            appPath = suggestedAppPath
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

    output = ""
    if cfg["venvCfg"]["venvEnabled"]:
        # todo - move to substitution rules
        for item in [
                {"src": cfg["venvCfg"]["venvName"], "dst": "venvName"},
                {"src": cfg["appPath"], "dst": "appPath"},
                {"src": sys.executable, "dst": "py"}
                ]:
            appCmd = CfgManager.singlestepStrFormat(
                appCmd,
                item["dst"],
                item["src"]
            )
        commitLogger.info("App command: {}".format(appCmd))
        # initialize venv
        p = subprocess.Popen('rm -rf {}'.format(
                cfg["venvCfg"]["venvDir"]
            ),
            shell=True,
            executable="/bin/bash"
        )
        p.wait()
        p.communicate()
        p = subprocess.Popen('mkdir {}'.format(
                cfg["venvCfg"]["venvDir"]
            ),
            shell=True,
            executable="/bin/bash"
        )
        p.wait()
        p.communicate()
        p = subprocess.Popen('{py} -m venv {venvName}'.format(
                py=sys.executable,
                venvName=cfg["venvCfg"]["venvName"]
            ),
            executable="/bin/bash",
            cwd=cfg["venvCfg"]["venvDir"],
            shell=True
        )
        p.wait()
        p.communicate()
        p = subprocess.Popen(
            appCmd,
            executable="/bin/bash",
            cwd=cfg["venvCfg"]["venvDir"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=newEnv,
            shell=True,
            encoding="utf-8", errors="replace"
        )
        for line in p.stdout:
            if cfg["verboseOutput"]:
                sys.stdout.write(line)
            commitLogger.info(line)
            output = "{}\n{}".format(output, line)
        p.wait()
        p = subprocess.Popen('rm -rf {}'.format(
                cfg["venvCfg"]["venvDir"]
            ),
            shell=True,
            executable="/bin/bash"
        )
        p.wait()
        p.communicate()
    else:
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
    if cfgData["cachedPathConfig"]["enabled"] == True:
        pathExists, cashedPath = getCashedPath(commit, cfgData)
        if pathExists:
            commitLogger.info(
                "Path, corresponding commit {c} is cashed, value:{p}".format(
                c=commit, p=cashedPath))
            if cfgData["cachedPathConfig"]["passCmdList"]:
                return
        else:
            commitLogger.info("Cfg = {}".format(cfgData))
            if cfgData["cachedPathConfig"]["scheme"] == "mandatory":
                commitLogger.info("Ignore commit {}".format(commit))
                raise BuildError(
                    errType=BuildError.BuildErrType.TO_IGNORE,
                    message="build error handled by skip",
                    commit=commit
                    )

    try:
        runCommandList(commit, cfgData)
        if cfgData["skipMode"]["flagSet"]["enableRebuild"]:
            cfgData["skipMode"]["flagSet"]["switchOnSimpleBuild"] = True
            cfgData["skipMode"]["flagSet"]["switchOnExtendedBuild"] = False
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
    cashMap = cfgData["cachedPathConfig"]["cashMap"]
    for k, v in cashMap.items():
        if shortHash in k:
            return True, cfgData["cachedPathConfig"]["cashMap"][k]
    return False, None


def getReducedInterval(list, cfg):
    # returns (True, reducedList) if given interval contains
    # two different prerun-cashed commits
    # [...[i1...<reduced interval>...i2]...]
    # and (False, None) otherwise
    if not cfg["cachedPathConfig"]["enabled"]:
        return False, None
    cashMap = cfg["cachedPathConfig"]["cashMap"]
    for i, commitHash in enumerate(list):
        list[i] = commitHash.replace('"', "")
    i1 = None
    i2 = None
    for commitHash in list:
        shortHash = getMeaningfullCommitTail(commitHash)
        for cashedCommit in cashMap:
            if shortHash in cashedCommit:
                i2 = commitHash
                break
    for commitHash in reversed(list):
        shortHash = getMeaningfullCommitTail(commitHash)
        for cashedCommit in cashMap:
            if shortHash in cashedCommit:
                i1 = commitHash
                break
    if i1 == i2:
        return False, None
    else:
        reducedList = []
        for i in list:
            if not reducedList:
                if i == i1:
                    reducedList.append(i)
            else:
                reducedList.append(i)
                if i == i2:
                    break
        return True, reducedList


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
    logFileName = os.path.join(logPath, logFileName)
    with open(logFileName, "w+"):  # clear old log
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


def runUtility(cfg, args):
    modName = args.utility
    try:
        mod = importlib.import_module(
            "utils.{un}".format(un=modName))
        utilName = checkAndGetUtilityByName(cfg, modName)
        utility = getattr(mod, utilName)
        utility(args)
    except ModuleNotFoundError as e:
        raise CfgError("No utility {} found".format(modName))


class CfgError(Exception):
    pass


class CashError(Exception):
    pass


class CmdError(Exception):
    pass


class PreliminaryAnalysisError(Exception):
    def __init__(self, message, errType):
        self.message = message
        self.errType = errType

    def __str__(self):
        return self.message

    class PreliminaryErrType(Enum):
        WRONG_COMMANDLINE = 0
        NO_DEGRADATION = 1
        UNSTABLE_APPLICATION = 2


class BuildError(Exception):
    class BuildErrType(Enum):
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


def checkAndGetUtilityByName(cfg, utilName):
    if not (utilName in cfg["utilMap"]):
        raise CfgError(
            "{utilName} is not registered in config".format(
                utilName=utilName
            )
        )
    else:
        return cfg["utilMap"][utilName]


def checkAndGetSubclass(clName, parentClass):
    cl = [cl for cl in parentClass.__subclasses__() if cl.__name__ == clName]
    if not (cl.__len__() == 1):
        raise CfgError("Class {clName} doesn't exist".format(clName=clName))
    else:
        return cl[0]


class DictHolder:
    def __init__(self, dict: dict = None):
        if dict is not None:
            for k, v in dict.items():
                setattr(self, k, v)


def formatJSON(content, formatLambda):
    if isinstance(content, dict):
        for k, value in content.items():
            content[k] = formatJSON(value, formatLambda)
    elif isinstance(content, list):
        for id, item in enumerate(content):
            content[id] = formatJSON(item, formatLambda)
    elif isinstance(content, str):
        content = formatLambda(content)
    else:
        # bool or digit object
        pass
    return content


def findJSONPathsByValue(content, soughtFor, curPath:list=['$']):
    pathVector = []
    if isinstance(content, dict):
        for k, value in content.items():
            newRes = findJSONPathsByValue(value, soughtFor,
                curPath.copy() + [k])
            if newRes:
                pathVector.extend(newRes)
    elif isinstance(content, list):
        for id, item in enumerate(content):
            newPath = curPath.copy()
            newPath.append('[{}]'.format(id))
            newRes = findJSONPathsByValue(item, soughtFor,
                newPath)
            if newRes:
                pathVector.extend(newRes)
    else:
        if content == soughtFor:
            pathVector.append(curPath)
    return pathVector


def findJSONPathsByKey(content, soughtFor, curPath:list=['$']):
    pathVector = []
    if isinstance(content, dict):
        for k, value in content.items():
            if k == soughtFor:
                pathVector.append(curPath)
            newRes = findJSONPathsByKey(value, soughtFor,
                curPath.copy() + [k])
            if newRes:
                pathVector.extend(newRes)
    elif isinstance(content, list):
        for id, item in enumerate(content):
            newPath = curPath.copy()
            newPath.append('[{}]'.format(id))
            newRes = findJSONPathsByKey(item, soughtFor,
                newPath)
            if newRes:
                pathVector.extend(newRes)
    else:
        pass
    return pathVector


def multiplyCfgByKey(content):
    substVector = getSubstVectorByKey(content)
    pathVector = findJSONPathsByKey(content, mulKey)
    n = -1
    for v in substVector:
        if n != -1 and len(v) != n:
            raise Exception("Inconsistent multiplication keys: sizes differ")
        n = len(v)
    substNum = len(substVector)
    retList = []
    for _ in range(n):
        retList.append(deepCopyJSON(content))
    for substIdx in range(substNum):
        for cfgIdx in range(n):
            deepMapUpdate(retList[cfgIdx], pathVector[substIdx][1:],
                          substVector[substIdx][cfgIdx])
    return retList


def deepCopyJSON(obj):
    return json.loads(json.dumps(obj, sort_keys=True))


def getSubstVectorByKey(content, soughtFor=mulKey):
    pathVector = findJSONPathsByKey(content, soughtFor)
    keyVector = []
    # populate configs
    def stepInto(jObj, item):
        try:
            idx = int(item[1:-1])
            if item[0] == '[' and item[-1] == ']':
                return jObj[idx]
        except ValueError:
            pass
        return jObj[item]
    for path in pathVector: # except root '$'
        curObj = content
        for item in path[1:]:
            curObj = stepInto(curObj, item)
        curObj = curObj[mulKey]
        if isinstance(curObj, list):
            keyVector.append(curObj)
        else:
            raise Exception("Wrong JSON type in multiplicaiton key")
    return keyVector


def applySubstitutionRules(cfg: map, rules: list, commit: str=None):
    # if commit is None or rule['type'] == 'static',
    # the rule is considered as static,
    # substitution proceeds as simple string replacing

    serviceCfg = cfg["serviceConfig"]
    if ("substRulesData" not in serviceCfg)\
        or ("dataChanged" not in serviceCfg["substRulesData"])\
        or (not serviceCfg["substRulesData"]["dataChanged"]):
        # create config copy for the first application of rules
        savedCfg = copy.deepcopy(cfg)
        serviceCfg["substRulesData"] = {
            "dataChanged": True,
            "savedCfg": savedCfg
        }
    else:
        # apply rules to the saved copy of config
        savedServiceCfg = copy.deepcopy(serviceCfg)
        for k in cfg:
            cfg[k] = copy.deepcopy(serviceCfg["substRulesData"]["savedCfg"][k])
        cfg["serviceConfig"] = savedServiceCfg

    for rule in rules:
        if not rule["enabled"]:
            continue
        # convert path to list and remove root symbol
        pathToSrc = rule["from"]
        pathToDst = rule["to"]
        pathToSrc = pathToSrc.split(".")
        pathToDst = pathToDst.split(".")
        pathToSrc.pop(0)
        pathToDst.pop(0)
        # setup positions for substitution
        srcPos, dstPos = cfg, cfg
        for item in pathToSrc:
            srcPos = srcPos[item]
        for item in pathToDst:
            dstPos = dstPos[item]
        ruleIsStatic = True if rule["type"] == "static" else False
        dstPos = formatJSON(
            dstPos,
            lambda content:
            CfgManager.singlestepStrFormat(
                content,
                rule["placeholder"],
                getMapValueByShortHash(srcPos, commit)\
                    if commit is not None and not ruleIsStatic\
                    else srcPos
            )
        )
        cfg = deepMapUpdate(cfg, pathToDst, dstPos)

def getMapValueByShortHash(map: dict, commit: str):
    for k in map:
        if getMeaningfullCommitTail(k) ==\
            getMeaningfullCommitTail(commit):
            return map[k]
    raise Exception("No {} in {}".format(
        commit, map.keys()
    ))

def deepMapUpdate(content: map, path: list, substitution):
    if not path:
        return substitution
    else:
        root = path.pop(0)
        if isinstance(content, list):
            try:
                root = int(root[1:-1])
            except TypeError:
                raise Exception("Wrong index {}".format(root))
        content[root] = deepMapUpdate(content[root], path, substitution)
        return content
    