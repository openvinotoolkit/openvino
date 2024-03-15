# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.pattern_match import apply_pattern


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
