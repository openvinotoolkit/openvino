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
from mo.front.extractor import add_output_ops
from mo.graph.graph import Graph


class OutputCut(FrontReplacementPattern):
    enabled = True
    run_not_recursively = True
    force_clean_up = True

    def run_after(self):
        from extensions.front.user_data_repack import UserDataRepack
        return [UserDataRepack]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        add_output_ops(graph, graph.graph['packed_outputs'], inputs=graph.graph['user_shapes'])
