#!/usr/bin/env python3
# ==============================================================================
#  Copyright 2017-2020 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
import argparse
import errno
import os
from subprocess import check_output, call
import sys
import shutil
import glob
import platform
import json
import shlex

def command_executor(cmd, verbose=False, msg=None, stdout=None):
    '''
    Executes the command.
    Example: 
      - command_executor('ls -lrt')
      - command_executor(['ls', '-lrt'])
    '''
    if type(cmd) == type([]):  #if its a list, convert to string
        cmd = ' '.join(cmd)
    if verbose:
        tag = 'Running COMMAND: ' if msg is None else msg
        print(tag + cmd)
    if (call(shlex.split(cmd), stdout=stdout) != 0):
        raise Exception("Error running command: " + cmd)

def download(target_name, repo, version):

    # First download to a temp folder
    call(["git", "clone", repo, target_name])

    # Next goto this folder nd determone the name of the root folder
    pwd = os.getcwd()

    # Go to the tree
    os.chdir(target_name)

    # checkout the specified branch
    command_executor(["git", "fetch"], verbose=True)
    command_executor(["git", "checkout", version], verbose=True)

    os.chdir(pwd)

# Get me the current sha for this commit
current_sha = check_output(['git', 'rev-parse', 'HEAD']).strip().decode("utf-8")
print("nGraph SHA: ", current_sha)

# Download ngraph-bridge 
download('ngraph-bridge', 'https://github.com/tensorflow/ngraph-bridge.git', 'master')

# Run ngraph-bridge-build
pwd = os.getcwd()
os.chdir('ngraph-bridge')
command_executor(['./build_ngtf.py', '--ngraph_version', current_sha])

# Now run the tests
os.environ['PYTHONPATH'] = os.getcwd() 
command_executor([
    'python3',
    'test/ci/buildkite/test_runner.py',
    '--artifacts',
    'build_cmake/artifacts',
    '--test_cpp'
])

os.chdir(pwd)
