"""
 Copyright (C) 2018-2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import glob
import os
import re
import subprocess
import sys
import numpy as np

def shell(cmd, env=None, cwd=None, out_format="plain"):
    """
    Run command execution in specified environment
    :param cmd: list containing command and its parameters
    :param env: set of environment variables to set for this command
    :param cwd: working directory from which execute call
    :param out_format: 'plain' or 'html'. If 'html' all '\n; symbols are replaced by '<br>' tag
    :return:
    """
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        cmd = ['/bin/bash', '-c', " ".join(cmd)]
    else:
        cmd = " ".join(cmd)

    sys.stdout.write("Running command:\n" + "".join(cmd) + "\n")
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    stdout = str(stdout.decode('utf-8'))
    stderr = str(stderr.decode('utf-8'))
    if out_format == "html":
        stdout = "<br>\n".join(stdout.split('\n'))
        stderr = "<br>\n".join(stderr.split('\n'))
    return p.returncode, stdout, stderr


def parse_avg_err(speech_sample_out):
    errors = []
    for line in speech_sample_out:
        if "avg error" in line:
            errors.append(float(line.split(': ')[1]))
    avg_error = round(np.mean(errors), 2)
    return avg_error


def fix_path(path, env_name, root_path=None):
    """
    Fix path: expand environment variables if any, make absolute path from
    root_path/path if path is relative, resolve symbolic links encountered.
    """
    path = os.path.expandvars(path)
    if not os.path.isabs(path) and root_path is not None:
        path = os.path.join(root_path, path)
    if env_name == "samples_data_zip":
        return path
    return os.path.realpath(os.path.abspath(path))


def fix_env_conf(env, root_path=None):
    """Fix paths in environment config."""
    for name, value in env.items():
        if isinstance(value, dict):
            # if value is dict, think of it as of a (sub)environment
            # within current environment
            # since it can also contain envvars/relative paths,
            # recursively update (sub)environment as well
            env[name] = fix_env_conf(value, root_path=root_path)
        else:
            env[name] = fix_path(value, name, root_path=root_path)
    return env