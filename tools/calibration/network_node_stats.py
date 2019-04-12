"""
Copyright (C) 2018-2019 Intel Corporation

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


class NetworkNodeStats:
    __slots__ = ['min_outputs', 'max_outputs']

    def __init__(self, channels_count: int):
        self.min_outputs = list()
        self.max_outputs = list()
        for i in range(channels_count):
            self.min_outputs.append(None)
            self.max_outputs.append(None)