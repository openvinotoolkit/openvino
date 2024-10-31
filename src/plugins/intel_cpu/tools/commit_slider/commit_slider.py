# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess
import os
import shutil
import sys
from distutils.dir_util import copy_tree
from distutils.errors import DistutilsFileError
from utils.cfg_manager import CfgManager
from utils.helpers import safeClearDir, getParams

args, cfgData, customCfgPath = getParams()

if args.utility != "no_utility":
    from utils.helpers import runUtility
    runUtility(cfgData, args)

elif args.isMultiply == "multiply":
    argString = " ".join(sys.argv)
    workPath = os.getcwd()
    formattedCmd = "{py} {argString} -x 'single'".format(
        py=sys.executable, argString=argString
    )
    subprocess.call(formattedCmd.split())

elif args.isWorkingDir:
    # rerun script from work directory
    from utils.modes import Mode
    from utils.helpers import CfgError
    from utils.helpers import checkArgAndGetCommits

    commitList = []
    if "commitList" in cfgData["runConfig"] and\
        "explicitList" in cfgData["runConfig"]["commitList"]:
            commitList = cfgData["runConfig"]["commitList"]["explicitList"]
    elif args.commitSeq is None:
        if "getCommitListCmd" in cfgData["runConfig"]["commitList"]:
            commitListCmd = cfgData["runConfig"]["commitList"]
            commitListCmd = commitListCmd["getCommitListCmd"]
            cwd = cfgData["gitPath"]
            try:
                out = subprocess.check_output(commitListCmd.split(), cwd=cwd)
            except subprocess.CalledProcessError as e:
                msg = "Commit list command caused error"
                raise CfgError("{msg} {e}".format(msg=msg, e=str(e)))
            out = out.decode("utf-8")
            commitList = out.split()
        elif "explicitList" in cfgData["runConfig"]["commitList"]:
            commitList = cfgData["runConfig"]["commitList"]["explicitList"]
        else:
            raise CfgError("Commit list is mandatory")
    else:
        commitList = checkArgAndGetCommits(args.commitSeq, cfgData)

    commitList.reverse()
    p = Mode.factory(cfgData)
    p.run(commitList, cfgData)
    p.printResult()

else:
    # prepare run
    workPath = cfgData["workPath"]
    if not os.path.exists(workPath):
        os.mkdir(workPath)
    else:
        safeClearDir(workPath, cfgData)
    curPath = os.getcwd()
    copy_tree(curPath, workPath)
    # handle user cache path
    tempCachePath = CfgManager.singlestepStrFormat(cfgData["cachePath"], "workPath", workPath)
    permCachePath = CfgManager.singlestepStrFormat(cfgData["cachePath"], "workPath", curPath)
    # setup cache path if specified
    if cfgData['userCachePath']:
        permCachePath = cfgData['userCachePath']
    else:
        cfgData['userCachePath'] = permCachePath

    # run CS
    scriptName = os.path.basename(__file__)
    argString = " ".join(sys.argv)
    formattedCmd = "{py} {workPath}/{argString} -wd".format(
        py=sys.executable, workPath=workPath, argString=argString
    )
    subprocess.call(formattedCmd.split())

    # copy logs and cache back to general repo
    tempLogPath = CfgManager.singlestepStrFormat(cfgData["logPath"], "workPath", workPath)
    permLogPath = CfgManager.singlestepStrFormat(cfgData["logPath"], "workPath", curPath)

    if cfgData['userLogPath']:
        permLogPath = cfgData['userLogPath']

    safeClearDir(permLogPath, cfgData)
    if not cfgData['clearLogsAposteriori']:
        copy_tree(tempLogPath, permLogPath)

    safeClearDir(permCachePath, cfgData)
    try:
        copy_tree(tempCachePath, permCachePath)
    except DistutilsFileError:
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

    safeClearDir(workPath, cfgData)
