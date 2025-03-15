# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess
import os
import shutil
import sys
from shutil import copytree
from utils.cfg_manager import CfgManager
from utils.helpers import safeClearDir, getParams, getActualCfg

args, cfgData, customCfgPath = getParams()

# multiconfig handling: fetch actual config by idx
curCfgData = getActualCfg(cfgData, args.multiconfig)

if isinstance(cfgData, list) and args.multiconfig == "undefined":
    for idx, _ in enumerate(cfgData):
        argString = " ".join(sys.argv)
        workPath = os.getcwd()
        formattedCmd = "{py} {argString} -x {idx}".format(
            py=sys.executable, argString=argString, idx=idx
        )
        subprocess.call(formattedCmd.split())
    exit()

if args.utility != "no_utility":
    from utils.helpers import runUtility
    runUtility(curCfgData, args)

elif args.isWorkingDir:
    # rerun script from work directory
    from utils.modes import Mode
    from utils.helpers import CfgError
    from utils.helpers import checkArgAndGetCommits

    commitList = []
    if "commitList" in curCfgData["runConfig"] and\
        "explicitList" in curCfgData["runConfig"]["commitList"]:
            commitList = curCfgData["runConfig"]["commitList"]["explicitList"]
    elif args.commitSeq is None:
        if "getCommitListCmd" in curCfgData["runConfig"]["commitList"]:
            commitListCmd = curCfgData["runConfig"]["commitList"]
            commitListCmd = commitListCmd["getCommitListCmd"]
            cwd = curCfgData["gitPath"]
            try:
                out = subprocess.check_output(commitListCmd.split(), cwd=cwd)
            except subprocess.CalledProcessError as e:
                msg = "Commit list command caused error"
                raise CfgError("{msg} {e}".format(msg=msg, e=str(e)))
            out = out.decode("utf-8")
            commitList = out.split()
        elif "explicitList" in curCfgData["runConfig"]["commitList"]:
            commitList = curCfgData["runConfig"]["commitList"]["explicitList"]
        else:
            raise CfgError("Commit list is mandatory")
    else:
        commitList = checkArgAndGetCommits(args.commitSeq, curCfgData)

    commitList.reverse()
    p = Mode.factory(curCfgData)
    p.run(commitList, curCfgData)
    p.printResult()

else:
    # prepare run
    workPath = curCfgData["workPath"]
    if not os.path.exists(workPath):
        os.mkdir(workPath)
    else:
        safeClearDir(workPath, curCfgData)
    curPath = os.getcwd()
    copytree(curPath, workPath, dirs_exist_ok=True)
    # handle user cache path
    tempCachePath = CfgManager.singlestepStrFormat(curCfgData["cachePath"], "workPath", workPath)
    permCachePath = CfgManager.singlestepStrFormat(curCfgData["cachePath"], "workPath", curPath)
    # setup cache path if specified
    if curCfgData['userCachePath']:
        permCachePath = curCfgData['userCachePath']
    else:
        curCfgData['userCachePath'] = permCachePath

    # run CS
    scriptName = os.path.basename(__file__)
    argString = " ".join(sys.argv)
    formattedCmd = "{py} {workPath}/{argString} -wd".format(
        py=sys.executable, workPath=workPath, argString=argString
    )
    subprocess.call(formattedCmd.split())

    # copy logs and cache back to general repo
    tempLogPath = CfgManager.singlestepStrFormat(curCfgData["logPath"], "workPath", workPath)
    permLogPath = CfgManager.singlestepStrFormat(curCfgData["logPath"], "workPath", curPath)

    if curCfgData['userLogPath']:
        permLogPath = curCfgData['userLogPath']

    if curCfgData['clearLogsAposteriori']:
        safeClearDir(permLogPath, curCfgData)
    elif not tempLogPath == permLogPath:
        safeClearDir(permLogPath, curCfgData)
        copytree(tempLogPath, permLogPath, dirs_exist_ok=True)

    safeClearDir(permCachePath, curCfgData)
    try:
        copytree(tempCachePath, permCachePath)
    except Exception:
        # prevent exception raising while cache is empty
        pass

    try:
        shutil.copyfile(
            os.path.join(workPath, customCfgPath),
            os.path.join(curPath, customCfgPath),
            follow_symlinks=True,
        )
    except shutil.SameFileError:
        # prevent exception raising if cfg set up from outer location
        pass

    safeClearDir(workPath, curCfgData)
