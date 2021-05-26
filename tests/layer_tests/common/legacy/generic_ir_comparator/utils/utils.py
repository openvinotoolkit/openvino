import subprocess
import sys


def shell(cmd, env=None, cwd=None):
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        cmd = ['/bin/bash', '-c', "".join(cmd)]
    else:
        cmd = " ".join(cmd)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    return p.returncode, stdout, stderr

