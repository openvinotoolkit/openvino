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


class Connection:
    def __init__(self, edge, port, layer):
        self._edge = edge
        self._port = port
        self._layer = layer

    @property
    def edge(self):
        return self._edge

    @property
    def port(self):
        return self._port

    @property
    def layer(self):
        return self._layer
