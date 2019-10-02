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


class Edge:
    def __init__(self, data: dict):
        self._from_layer = int(data['from-layer'])
        self._from_port = int(data['from-port'])
        self._to_layer = int(data['to-layer'])
        self._to_port = int(data['to-port'])

    @property
    def from_layer(self) -> int:
        return self._from_layer

    @property
    def from_port(self) -> int:
        return self._from_port

    @property
    def to_layer(self) -> int:
        return self._to_layer

    @property
    def to_port(self) -> int:
        return self._to_port
