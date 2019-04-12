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
from mo.front.common.custom_replacement_registry import CustomReplacementRegistry
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.replacement import FrontReplacementFromConfigFileOp
from mo.graph.graph import Graph
from mo.utils.class_registration import update_registration, get_enabled_and_disabled_transforms


class TensorflowUseCustomOperationsConfig(FrontReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].tensorflow_use_custom_operations_config is not None]

    def run_before(self):
        from extensions.front.pass_separator import FrontStart
        return [FrontStart]

    def run_after(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        argv = graph.graph['cmd_params']
        registry = CustomReplacementRegistry()
        registry.add_custom_replacement_description_from_config(argv.tensorflow_use_custom_operations_config)

        # automatically generate sub-classes for custom replacements that replace sub-graph with a single node
        for replacement_desc in registry.get_all_replacements_descriptions():
            if replacement_desc.has('op'):
                type('FrontReplacementFromConfigFileOp' + replacement_desc.op, (FrontReplacementFromConfigFileOp,),
                     {'replacement_id': replacement_desc.id})
        update_registration([FrontReplacementFromConfigFileOp], *get_enabled_and_disabled_transforms())
