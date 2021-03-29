# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class BackStart(BackReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.ApplyPermutations import ApplyPermutation
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
