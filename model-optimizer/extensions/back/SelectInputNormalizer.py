"""
 Copyright (c) 2019 Intel Corporation

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

from extensions.back.ReshapeMutation import ReshapeMutation
from extensions.back.SelectBroadcast import SelectBroadcast
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.unsqueeze import Unsqueeze


class SelectInputNormalizer(BackReplacementPattern):
    """
        Temporary workaround (while 0D isn't supported).
        Select in TF can have 0D input condition.
    """
    enabled = True

    def run_before(self):
        return [ReshapeMutation, SelectBroadcast]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', op='Select'))],
            edges=[]
        )

    op_list = ['ReduceAnd']

    def replace_pattern(self, graph: Graph, match: dict):
        select = match['op']
        if select.has_valid('format') and select['format'] == 'tf':
            condition = select.in_port(0)

            if condition.data.get_shape().size == 0:
                # Some checks that we can reshape
                input_condition = condition.get_connection().get_source().node
                if input_condition.op in self.op_list:
                    condition.data.set_shape(np.array([1]))
                    if condition.data.get_value() is not None:
                        condition.data.set_value(np.array([condition.data.get_value()]))
