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


class NchwShape:
    def __init__(self, n: int, c: int, h: int, w: int):
        self._n = n
        self._c = c
        self._h = h
        self._w = w

    @property
    def layout(self) -> str:
        return 'NCHW'

    @property
    def n(self) -> int:
        return self._n

    @property
    def c(self) -> int:
        return self._c

    @property
    def h(self) -> int:
        return self._h

    @property
    def w(self) -> int:
        return self._w


class ChwShape:
    def __init__(self, c: int, h: int, w: int):
        self._c = c
        self._h = h
        self._w = w

    @property
    def n(self) -> int:
        return 1

    @property
    def layout(self) -> str:
        return 'CHW'

    @property
    def c(self) -> int:
        return self._c

    @property
    def h(self) -> int:
        return self._h

    @property
    def w(self) -> int:
        return self._w


class NcShape:
    def __init__(self, n: int, c: int):
        self._n = n
        self._c = c

    @property
    def layout(self) -> str:
        return 'NC'

    @property
    def n(self) -> int:
        return self._n

    @property
    def c(self) -> int:
        return self._c


class CShape:
    def __init__(self, c: int):
        self._n = 1
        self._c = c

    @property
    def layout(self) -> str:
        return 'C'

    @property
    def n(self) -> int:
        return self._n

    @property
    def c(self) -> int:
        return self._c


class Shape:
    @staticmethod
    def create(layout:str, dims):
        if layout == 'NCHW':
            return NchwShape(dims[0], dims[1], dims[2], dims[3])
        if layout == 'CHW':
            return ChwShape(dims[0], dims[1], dims[2])
        elif layout == 'NC':
            return NcShape(dims[0], dims[1])
        elif layout == 'C':
            return CShape(dims[0])
        else:
            raise ValueError("not supported layout '{}'".format(layout))
