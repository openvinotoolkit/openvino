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
from extensions.middle.fusings import Fusing
from extensions.middle.pass_separator import PostMiddleStart
from mo.graph.graph import Graph
from mo.middle.passes.leaky_relu import convert_mul_eltwise_to_leaky_relu
from mo.middle.replacement import MiddleReplacementPattern


class LeakyReLU(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        return [Fusing]

    def run_before(self):
        return [PostMiddleStart]

    def find_and_replace_pattern(self, graph: Graph):
        convert_mul_eltwise_to_leaky_relu(graph)
