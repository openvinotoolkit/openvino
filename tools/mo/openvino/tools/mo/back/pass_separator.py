# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph


class BackStart(BackReplacementPattern):
    enabled = True

    def run_after(self):
        from openvino.tools.mo.middle.ApplyPermutations import ApplyPermutation
        return [ApplyPermutation]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        pass


class BackFinish(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        return []

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        pass
