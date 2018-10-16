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

from mo.graph.graph import Node


def roipooling_infer(node: Node):
    """
    Sets shape of output node according specified parameters input blobs and node
    Sets number from the first input blob, channels from the second one, height and width are specified
    Parameters
    ----------
    node
    """
    shapes = [node.in_node(i).shape for i in range(len(node.in_nodes()))]
    if any(s is None for s in shapes):
        return

    num = shapes[1][0]
    height = node.pooled_h
    width = node.pooled_w
    if node.has_valid('framework') and node['framework'] == 'tensorflow':
        channels = shapes[0][3]
        node.out_node().shape = np.array([num, height, width, channels])
    else:
        channels = shapes[0][1]
        node.out_node().shape = np.array([num, channels, height, width])
