"""
 Copyright (c) 2018-2019 Intel Corporation

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

import networkx as nx

from mo.graph.graph import Graph
from mo.middle.pattern_match import apply_pattern


class ReplacementPattern(object):
    # List of classes that shouldn't be treated as standalone replacers
    # All intermediate infrastructure classes should be here
    excluded_replacers = []

    def find_and_replace_pattern(self, graph: Graph):
        apply_pattern(graph, **self.pattern(), action=self.replace_pattern)  # pylint: disable=no-member

    def run_before(self):
        """
        Returns list of replacer classes which this replacer must be run before.
        :return: list of classes
        """
        return []

    def run_after(self):
        """
        Returns list of replacer classes which this replacer must be run after.
        :return: list of classes
        """
        return []
