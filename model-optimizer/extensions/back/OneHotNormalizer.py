"""
 Copyright (C) 2018-2020 Intel Corporation

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
import logging as log

import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import data_type_str_to_np


class OneHotNormalizer(BackReplacementPattern):
    """
    The transformation converts the OneHot layer to V7 specification:
    1. The inputs with on/off values are converted to attributes "on_value" and "off_value".
    2. The input with depth value is converted to attribute "depth".
    3. The output data type of the layer is inferred from the data type of on/off value taking into account IE supported
    types.
    """
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def find_and_replace_pattern(self, graph: Graph):
        assert_msg = "OneHot `{0}` ({1} input port value) should be scalar: node: `{2}`, {0} value: `{3}`"
        for node in graph.get_op_nodes(type='OneHot'):
            depth = node.in_port(1).data.get_value()
            assert depth is not None and depth.ndim == 0, assert_msg.format('depth', '1', node.name, depth)
            depth = depth.item(0)

            on_value = node.in_port(2).data.get_value()
            assert on_value is not None and on_value.ndim == 0, assert_msg.format('on_value', '2', node.name, on_value)
            off_value = node.in_port(3).data.get_value()
            assert off_value is not None and off_value.ndim == 0, assert_msg.format('off_value', '3', node.name,
                                                                                    off_value)

            if on_value.dtype in [np.int64, np.bool]:
                log.warning('Converting on_value and off_value to int32')
                on_value = np.int32(on_value.item(0))
                off_value = np.int32(off_value.item(0))
                node.data_type = np.int32
            if on_value.dtype == np.float32 and data_type_str_to_np(node.graph.graph['cmd_params'].data_type) == np.float16:
                log.warning('Converting on_value and off_value to fp16')
                on_value = np.float16(on_value.item(0))
                off_value = np.float16(off_value.item(0))
                node.data_type = np.float16
            else:
                node.data_type = on_value.dtype
                on_value = on_value.item(0)
                off_value = off_value.item(0)

            node['depth'] = depth
            node['on_value'] = on_value
            node['off_value'] = off_value

            node.in_port(1).disconnect()
            node.in_port(2).disconnect()
            node.in_port(3).disconnect()
