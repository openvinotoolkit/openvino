"""
 Copyright (C) 2021 Intel Corporation

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
from mo.graph.graph import Graph


class MemoryOffsetBatchUpdate(FrontReplacementPattern):
    enabled = True
    run_not_recursively = True

    def run_after(self):
        from extensions.front.user_data_repack import UserDataRepack
        return [UserDataRepack]

    def run_before(self):
        from extensions.front.kaldi.split_recurrent_memoryoffset import SplitRecurrentMemoryOffset
        return [SplitRecurrentMemoryOffset]

    def find_and_replace_pattern(self, graph: Graph):
        batch = graph.get_op_nodes(op="Parameter")[0].shape[0]
        memoryoffset_nodes = graph.get_op_nodes(op='MemoryOffset')
        for memoryoffset_node in memoryoffset_nodes:
            if memoryoffset_node.has_valid('element_size'):
                memoryoffset_node.element_size[0] = batch
