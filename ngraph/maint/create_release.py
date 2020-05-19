# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import sys
import os
import subprocess
import tempfile
import tarfile
import zipfile

assert sys.version_info >= (3, 4)
swd = os.path.dirname(os.path.realpath(__file__))
swd = swd + '/..'
swd = os.path.realpath(swd)
get_tag = subprocess.Popen(['git', 'describe', '--tags', '--abbrev=0', '--match', 'v*.*.*'],
                           cwd=swd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
tag = [line.strip().decode() for line in get_tag.stdout.readlines()]
retval = get_tag.wait()

get_files = subprocess.Popen(['git', 'ls-tree', '-r', 'HEAD', '--name-only'],
                             cwd=swd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

files = [line.strip().decode() for line in get_files.stdout.readlines()]
retval = get_files.wait()

# create tarball for Linux and macOS
with tarfile.open(swd + '/ngraph-' + tag[0] + '.tar.gz', 'w:gz') as tar:
    for f in files:
        file_path = swd + '/' + f
        tar.add(file_path, arcname='ngraph-' + tag[0][1:] + '/' + f)
    with tempfile.NamedTemporaryFile() as tag_file:
        tag_line = tag[0] + '\n'
        tag_file.write(tag_line.encode())
        tag_file.flush()
        tar.add(tag_file.name, arcname='ngraph-' + tag[0][1:] + '/TAG')

# create zipfile for Windows
TO = b'\r\n'
FROM = b'\n'

with zipfile.ZipFile(swd + '/ngraph-' + tag[0] + '.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    for f in files:
        with open(file_path, 'rb') as unix_file:
            win_content = unix_file.read().replace(FROM, TO)
        with tempfile.NamedTemporaryFile() as win_file:
            win_file.write(win_content)
            zipf.write(win_file.name, arcname='ngraph-' + tag[0][1:] + '/' + f)
    with tempfile.NamedTemporaryFile() as tag_file:
        tag_line = tag[0] + '\r\n'
        tag_file.write(tag_line.encode())
        tag_file.flush()
        zipf.write(tag_file.name, arcname='ngraph-' + tag[0][1:] + '/TAG')
