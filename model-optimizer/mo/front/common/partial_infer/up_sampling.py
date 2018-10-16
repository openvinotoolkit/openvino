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

import numpy as np


def up_sampling_infer(node):
    if node.scale is None:
        return
    input_shape = node.in_node(0).shape
    batch = input_shape[0]
    channel = input_shape[1]
    y = input_shape[2] * node.scale
    x = input_shape[3] * node.scale
    node.out_node(0).shape = np.array([batch, channel, y, x])
