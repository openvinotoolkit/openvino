import subprocess
import os
import shutil
import sys
from distutils.dir_util import copy_tree
from utils.helpers import safeClearDir, getParams

args, cfgData = getParams()

if args.__dict__["isWorkingDir"]:
    # rerun script from work directory
    from utils.modes import Mode
    from utils.helpers import CfgError
    from utils.helpers import checkArgAndGetCommitList

    commitList = []
    if (args.__dict__["commitSeq"] == None):
        if ("getCommitListCmd" in cfgData["specialConfig"]["commitList"]):
            commitListCmd = cfgData["specialConfig"]["commitList"]["getCommitListCmd"]
            cwd = cfgData["commonConfig"]["gitPath"]
            try:
                out = subprocess.check_output(commitListCmd.split(), cwd=cwd)
            except subprocess.CalledProcessError as e:
                raise CfgError("Commit list command caused error {}".format(str(e)))
            out = out.decode('utf-8')
            commitList = out.split()
        else:
            raise CfgError("Commit list is mandatory")
    else:
        commitList = checkArgAndGetCommitList(args.__dict__["commitSeq"], cfgData)

    commitList.reverse()
    p = Mode.factory(cfgData)
    p.run(0, len(commitList) - 1, commitList, cfgData)
    p.getResult()

else:
    workPath = cfgData["commonConfig"]["workPath"]
    if not os.path.exists(workPath):
        os.mkdir(workPath)
    else:
        safeClearDir(workPath)
    curPath = os.getcwd()
    copy_tree(curPath, workPath)
    scriptName = os.path.basename(__file__)
    argString = ' '.join(sys.argv)
    formattedCmd = "{py} {workPath}/{argString} -wd".format(
        py=sys.executable,
        workPath=workPath,
        argString=argString)
    subprocess.call(formattedCmd.split())

    # copy logs and cache back to general repo
    tempLogPath = cfgData["commonConfig"]["logPath"].format(workPath=workPath)
    permLogPath = cfgData["commonConfig"]["logPath"].format(workPath=curPath)
    safeClearDir(permLogPath)
    copy_tree(tempLogPath, permLogPath)

    tempCachePath = cfgData["commonConfig"]["cachePath"].format(workPath=workPath)
    permCachePath = cfgData["commonConfig"]["cachePath"].format(workPath=curPath)
    safeClearDir(permCachePath)
    copy_tree(tempCachePath, permCachePath)

    cfgPath = 'utils/cfg.json'
    shutil.copyfile(os.path.join(workPath, cfgPath), os.path.join(curPath, cfgPath), follow_symlinks=True)
    safeClearDir(workPath)
