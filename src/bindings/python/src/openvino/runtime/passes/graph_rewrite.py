# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.pyopenvino.passes import GraphRewrite as GraphRewriteBase


class GraphRewrite(GraphRewriteBase):
    """GraphRewrite that additionally holds python transformations objects."""
    def __init__(self):
        super().__init__()
        self.passes_list = []  # need to keep python instances alive

    def add_matcher(self, transformation):
        self.passes_list.append(transformation)
        return super().add_matcher(transformation)
