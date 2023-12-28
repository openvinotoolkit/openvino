import subprocess
import os
import shutil
import sys
from distutils.dir_util import copy_tree
from utils.helpers import safeClearDir, getParams

args, cfgData, customCfgPath = getParams()

if args.__dict__["isWorkingDir"]:
    # rerun script from work directory
    from utils.modes import Mode
    from utils.helpers import CfgError
    from utils.helpers import checkArgAndGetCommits

    commitList = []
    if args.__dict__["commitSeq"] is None:
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
        commitList = checkArgAndGetCommits(args.__dict__["commitSeq"], cfgData)

    commitList.reverse()
    p = Mode.factory(cfgData)
    p.run(commitList, cfgData)
    p.printResult()

else:
    workPath = cfgData["workPath"]
    if not os.path.exists(workPath):
        os.mkdir(workPath)
    else:
        safeClearDir(workPath, cfgData)
    curPath = os.getcwd()
    copy_tree(curPath, workPath)
    scriptName = os.path.basename(__file__)
    argString = " ".join(sys.argv)
    formattedCmd = "{py} {workPath}/{argString} -wd".format(
        py=sys.executable, workPath=workPath, argString=argString
    )
    subprocess.call(formattedCmd.split())

    # copy logs and cache back to general repo
    tempLogPath = cfgData["logPath"].format(workPath=workPath)
    permLogPath = cfgData["logPath"].format(workPath=curPath)
    safeClearDir(permLogPath, cfgData)
    copy_tree(tempLogPath, permLogPath)

    tempCachePath = cfgData["cachePath"].format(workPath=workPath)
    permCachePath = cfgData["cachePath"].format(workPath=curPath)
    safeClearDir(permCachePath, cfgData)
    copy_tree(tempCachePath, permCachePath)

    shutil.copyfile(
        os.path.join(workPath, customCfgPath),
        os.path.join(curPath, customCfgPath),
        follow_symlinks=True,
    )
    safeClearDir(workPath, cfgData)
