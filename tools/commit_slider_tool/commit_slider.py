import subprocess
from utils.modes import Mode
from utils.helpers import CfgError
from utils.helpers import checkArgAndGetCommitList
from argparse import ArgumentParser
import json

parser = ArgumentParser()
parser.add_argument("-c", "--commits", dest="commitSeq", help="commit sequence")
parser.add_argument("-cfg", "--config", dest="configuration", help="configuration source")
args = parser.parse_args()

cfgPath = ""
if (args.__dict__["configuration"] == None):
    cfgPath = 'utils/cfg.json'
else:
    cfgPath = args.__dict__["configuration"]
cfgData = json.load(open(cfgPath))

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
