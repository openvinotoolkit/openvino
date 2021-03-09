"""
 Copyright (C) 2018-2021 Intel Corporation

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

from mo.middle.passes.convert_data_type import destination_type_to_np_data_type
from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender


class Convert_extender(Extender):
    op = 'Convert'

    @staticmethod
    def extend(op: Node):
        op['dst_type'] = destination_type_to_np_data_type(op.destination_type)
        # CompressQuantizeWeights generates IR with constant sub-graph, that should not be ConstFolded:
        #   Const(u8) -> Convert(to fp) -> (some eltwise operations) -> FakeQuantize
        if op.in_node().in_node().soft_get('type') == 'Const':
            op['stop_value_propagation'] = True
