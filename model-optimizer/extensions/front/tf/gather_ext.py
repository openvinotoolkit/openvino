# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.gather import Gather, AttributedGather
from mo.front.extractor import FrontExtractorOp


class GatherFrontExtractor(FrontExtractorOp):
    op = 'Gather'
    enabled = True

    @classmethod
    def extract(cls, node):
        AttributedGather.update_node_stat(node, {'axis': 0})
        return cls.enabled


class ResourceGatherFrontExtractor(FrontExtractorOp):
    op = 'ResourceGather'
    enabled = True

    @classmethod
    def extract(cls, node):
        AttributedGather.update_node_stat(node, {'axis': 0})
        return cls.enabled


class GatherV2FrontExtractor(FrontExtractorOp):
    op = 'GatherV2'
    enabled = True

    @classmethod
    def extract(cls, node):
        Gather.update_node_stat(node, {'batch_dims': node.pb.attr['batch_dims'].i})
        return cls.enabled
