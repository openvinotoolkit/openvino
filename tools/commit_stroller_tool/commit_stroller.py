import subprocess
from utils.modes import Mode
pr = subprocess.Popen(['git', 'checkout', 'master'], cwd="samples/sample3/",
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
pr.wait()
out = subprocess.check_output(['git', 'log', '-6', #  '297f..0b67', 
 '--pretty=format:"%h"'], cwd="samples/sample3/")
commitList = out.split()
commitList.reverse()
cfgFile = open('utils/cfg.json')
import json
cfgData = json.load(cfgFile)
p = Mode.factory(cfgData)
print ("commit found: {c}".format(c=commitList[p.run(0, len(commitList) - 1, commitList, cfgData)].decode('utf-8')))