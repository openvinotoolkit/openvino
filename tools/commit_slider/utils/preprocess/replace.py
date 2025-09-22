import re
import fileinput
import os
from utils.helpers import CfgError

def replace(cfg, commit=None):
    prepCfg = cfg["runConfig"]["preprocess"]
    filePath = prepCfg["file"]
    curDir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../../../../"))
    filePath = os.path.join(curDir, filePath)
    pattern = prepCfg["pattern"]
    replacement = ''
    if "replacement" in prepCfg:
        replacement = prepCfg["replacement"]
    for line in fileinput.input(filePath, inplace=True):
        newLine = re.sub(pattern, r'{}'.format(replacement), line, flags=0)
        print(newLine, end='')

# example: python3 -m commit_slider -u replace -file src/plugins/intel_cpu/src/graph.cpp  -pattern "\!node->isDynamicNode\(\)\ &&\ \!node->isExecutable\(\)\ &&\ \!node->isInPlace\(\)" -replacement "false"
def replacePreprocess(args):
    argDict = vars(args)
    if "-file" not in argDict:
        raise CfgError("No 'file' for replace-pp provided")
    if "-pattern" not in argDict:
        raise CfgError("No 'pattern' for replace-pp provided")

    tmpCfg = { "runConfig": { "preprocess" : {
        "file" : argDict["-file"],
        "pattern" : argDict["-pattern"],
        "replacement" : argDict["-replacement"] if "-replacement" in argDict else ''
    }}}
    replace(tmpCfg)
