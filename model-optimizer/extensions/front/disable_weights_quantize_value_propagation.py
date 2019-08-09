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
from extensions.front.tf.FakeQuantWithMinMaxVars import FakeQuantWithMinMaxVarsToQuantize
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph


class DisableQuantizeValuePropagation(FrontReplacementPattern):
    enabled = True
    graph_condition = [
        lambda graph: graph.graph['cmd_params'].keep_quantize_ops_in_IR
                      and graph.graph['cmd_params'].blobs_as_inputs
                      or graph.graph['cmd_params'].generate_experimental_IR_V10
    ]

    def run_after(self):
        return [FakeQuantWithMinMaxVarsToQuantize]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('quantize', dict(op='FakeQuantize', levels=lambda levels: levels != 2)),
                ('conv', dict(type=lambda type: type in ['Convolution', 'FullyConnected', 'MatMul'])),
            ],
            edges=[('quantize', 'conv', {'in': 1})]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        match['quantize']['stop_value_propagation'] = True
