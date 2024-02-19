# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.op import Op


class FakeQuantWithMinMaxVarsExtractor(FrontExtractorOp):
    op = 'FakeQuantWithMinMaxVars'
    enabled = True

    @classmethod
    def extract(cls, node):
        narrow_range = node.pb.attr['narrow_range'].b
        num_bits = node.pb.attr['num_bits'].i
        levels = 2 ** num_bits - int(narrow_range)

        # we prepare this operation to be converted to FakeQuantize op,
        # but input reconnection is needed, so we don't set infer function and type attribute
        Op.update_node_stat(node, {'op': 'FakeQuantWithMinMaxVars', 'levels': levels,
                                   'narrow_range': narrow_range, 'num_bits': num_bits})

        return cls.enabled


class FakeQuantWithMinMaxVarsPerChannelExtractor(FrontExtractorOp):
    op = 'FakeQuantWithMinMaxVarsPerChannel'
    enabled = True

    @classmethod
    def extract(cls, node):
        narrow_range = node.pb.attr['narrow_range'].b
        num_bits = node.pb.attr['num_bits'].i
        levels = 2 ** num_bits - int(narrow_range)

        # we prepare this operation to be converted to FakeQuantize op,
        # but input reconnection is needed, so we don't set infer function and type attribute
        Op.update_node_stat(node, {'op': 'FakeQuantWithMinMaxVars', 'levels': levels,
                                   'narrow_range': narrow_range, 'num_bits': num_bits})

        return cls.enabled
