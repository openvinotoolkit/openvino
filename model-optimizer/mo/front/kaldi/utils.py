"""
 Copyright (c) 2018 Intel Corporation

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

import struct


def get_uint16(s):
    return struct.unpack('H', s)[0]


def get_uint32(s):
    return struct.unpack('I', s)[0]


class KaldiNode:
    def __init__(self, name):
        self.name = name
        self.blobs = [None, None]

    def set_weight(self, w):
        self.blobs[0] = w

    def set_bias(self, b):
        self.blobs[1] = b

    def set_attrs(self, attrs: dict):
        for k, v in attrs.items():
            setattr(self, k, v)
