################################################################################
# Copyright 2021 Intel Corporation
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
################################################################################

import sys

status = {'SUCCESS': 0, 'FAILED': 1}


def get_version():
    version = sys.version.split(' ')[0].split('.')
    return {
        'major': int(version[0]),
        'minor': int(version[1]),
        'fix': int(version[2])
    }

def check_version():
    v = get_version()
    if not (v['major'] >= 3 and v['minor'] >= 7):
        print("ERROR: unsupported python version")
        return status.get('FAILED')
    return status.get('SUCCESS')
