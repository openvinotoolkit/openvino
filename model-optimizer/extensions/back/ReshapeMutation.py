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
from extensions.back.ForceStrictPrecision import ForceStrictPrecision
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class ReshapeMutation(BackReplacementPattern):
    enabled = True
    force_clean_up = True
    run_not_recursively = True

    def run_before(self):
        return [ForceStrictPrecision]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('reshape', {'kind': 'op'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        reshape = match['reshape']

        if reshape.soft_get('type') == 'Reshape':
            reshape['force_precision_in_ports'] = {1: 'int64'}
