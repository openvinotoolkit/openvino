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
import numpy as np

from extensions.back.ForceStrictPrecision import ForceStrictPrecision
from extensions.ops.prelu import PReLU
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, rename_node
from mo.ops.const import Const


class LeakyReLUMutation(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_before(self):
        return [ForceStrictPrecision]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('leakyrelu', dict(kind='op', op='LeakyReLU'))],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        relu = match['leakyrelu']
        relu_name = relu.soft_get('name', relu.id)
        if not relu.has_valid('negative_slope'):
            return

        rename_node(relu, relu_name + '/to_delete')
        # Create PReLU op and reconnect input/output from LeakyReLU to PReLU
        prelu = PReLU(graph, dict(name=relu_name)).create_node()
        rename_node(prelu, relu_name)

        const = Const(graph, dict(name=relu_name + "/weights", value=np.array([relu.negative_slope]))).create_node()

        relu.in_port(0).get_connection().set_destination(prelu.in_port(0))
        const.out_port(0).connect(prelu.in_port(1))
        relu.out_port(0).get_connection().set_source(prelu.out_port(0))
