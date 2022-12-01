import subprocess
from argparse import ArgumentParser
import os
import sys
from distutils.dir_util import copy_tree
from utils.helpers import getSafeClearDirProc
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

# rerun script from work directory
workPath = cfgData["commonConfig"]["workPath"]
isWorkingDir = args.__dict__["isWorkingDir"]
if not isWorkingDir:
    if not os.path.exists(workPath):
        os.mkdir(workPath)
    else:
        p = getSafeClearDirProc(workPath)
        p.wait()
    curPath = os.getcwd()
    copy_tree(curPath, workPath)
    scriptName = os.path.basename(__file__)
    argString = ' '.join(sys.argv)
    str = "python3.8 {workPath}/{argString} -wd true".format(workPath=workPath,
        argString=argString)
    os.system(str)
    # copy logs back
    tempLogPath = cfgData["commonConfig"]["logPath"].format(workPath=workPath)
    permLogPath = cfgData["commonConfig"]["logPath"].format(workPath=curPath)
    p = getSafeClearDirProc(permLogPath)
    p.wait()
    copy_tree(tempLogPath, permLogPath)
    p = getSafeClearDirProc(workPath)
    p.wait()
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
print ("Commit found: {c}".format(c=commitList[p.run(0, len(commitList) - 1, commitList, cfgData)]))
