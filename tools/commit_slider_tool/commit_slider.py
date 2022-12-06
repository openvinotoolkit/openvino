import subprocess
from argparse import ArgumentParser
import os
import shutil
import sys
from distutils.dir_util import copy_tree
from utils.helpers import safeClearDir, absolutizePaths
import json

parser = ArgumentParser()
parser.add_argument("-c", "--commits", dest="commitSeq", help="commit sequence")
parser.add_argument("-cfg", "--config", dest="configuration", help="configuration source")
parser.add_argument("-wd", "--workdir", type=bool, dest="isWorkingDir", default=False, help="flag if current directory is working")
args = parser.parse_args()

cfgPath = ""
if (args.__dict__["configuration"] == None):
    cfgPath = 'utils/cfg.json'
else:
    cfgPath = args.__dict__["configuration"]
cfgData = json.load(open(cfgPath))
cfgData = absolutizePaths(cfgData)

# rerun script from work directory
workPath = cfgData["commonConfig"]["workPath"]
isWorkingDir = args.__dict__["isWorkingDir"]
if not isWorkingDir:
    if not os.path.exists(workPath):
        os.mkdir(workPath)
    else:
        safeClearDir(workPath)
    curPath = os.getcwd()
    copy_tree(curPath, workPath)
    scriptName = os.path.basename(__file__)
    argString = ' '.join(sys.argv)
    formattedCmd = "python3 {workPath}/{argString} -wd true".format(
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
    # safeClearDir(workPath)
    exit()

# prevent cross import
from utils.modes import Mode
from utils.helpers import CfgError
from utils.helpers import checkArgAndGetCommitList

commitList = []
if (args.__dict__["commitSeq"] == None):
    if ("getCommitListCmd" in cfgData["specialConfig"]["commitList"].keys()):
        commitListCmd = cfgData["specialConfig"]["commitList"]["getCommitListCmd"]
        cwd = cfgData["commonConfig"]["gitPath"]
        out = subprocess.check_output(commitListCmd.split(), cwd=cwd)
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