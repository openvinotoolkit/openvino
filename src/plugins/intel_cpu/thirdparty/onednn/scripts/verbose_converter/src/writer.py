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


class Writer:
    def __init__(self, verbose_level=0):
        self.__verbose_level = int(verbose_level)
        self.__file = None

    def print(self, string, type):
        if type == 'WARN':
            print(f"{type}: {string}")
        if type == 'INFO':
            if self.__verbose_level > 0:
                print(string)
        if type == 'STDIO':
            print(string)
