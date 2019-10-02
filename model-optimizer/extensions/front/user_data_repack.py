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
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.extractor import user_data_repack
from mo.graph.graph import Graph


class UserDataRepack(FrontReplacementPattern):
    enabled = True
    run_not_recursively = True

    def run_after(self):
        return []

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        argv = graph.graph['cmd_params']

        packed_user_shapes, packed_outputs, freeze_placeholder = user_data_repack(
            graph, argv.placeholder_shapes, argv.output, argv.freeze_placeholder_with_value)

        graph.graph['user_shapes'] = packed_user_shapes
        graph.graph['packed_outputs'] = packed_outputs
        graph.graph['freeze_placeholder'] = freeze_placeholder

        inputs = list(packed_user_shapes.keys()) \
            if packed_user_shapes is not None and isinstance(packed_user_shapes, dict) else None
        graph.graph['inputs'] = inputs  # save user defined inputs for other extensions
