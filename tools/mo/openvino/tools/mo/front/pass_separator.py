# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph


class FrontStart(FrontReplacementPattern):
    enabled = True

    def run_after(self):
        return []

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        pass


class FrontFinish(FrontReplacementPattern):
    enabled = True

    def run_after(self):
        return []

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        pass
