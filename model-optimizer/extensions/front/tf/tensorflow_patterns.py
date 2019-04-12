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
import mo.front.tf.custom_subgraph_call as csc
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph


class TensorflowSubgraphPatterns(FrontReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].tensorflow_subgraph_patterns is not None]

    def run_before(self):
        return []

    def run_after(self):
        from extensions.front.tf.tensorflow_custom_operations_config_update import \
            TensorflowCustomOperationsConfigUpdate
        return [TensorflowCustomOperationsConfigUpdate]

    def find_and_replace_pattern(self, graph: Graph):
        argv = graph.graph['cmd_params']
        csc.replace_subgraph_calls(graph, argv.tensorflow_subgraph_patterns)


class TensorflowOperationPatterns(FrontReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].tensorflow_operation_patterns is not None]

    def run_before(self):
        from extensions.front.tf.tensorflow_use_custom_operations_config import TensorflowUseCustomOperationsConfig
        return [TensorflowUseCustomOperationsConfig]

    def run_after(self):
        return [TensorflowSubgraphPatterns]

    def find_and_replace_pattern(self, graph: Graph):
        argv = graph.graph['cmd_params']
        csc.offload_operations_to_tf(graph, argv.tensorflow_operation_patterns)
