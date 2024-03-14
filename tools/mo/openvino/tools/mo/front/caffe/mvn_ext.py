# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.mvn import MVNCaffe
from openvino.tools.mo.front.caffe.collect_attributes import collect_attributes
from openvino.tools.mo.front.extractor import FrontExtractorOp


class MVNFrontExtractor(FrontExtractorOp):
    op = 'MVN'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.mvn_param

        attrs = collect_attributes(param)

        if 'normalize_variance' not in attrs:
            attrs['normalize_variance'] = 1
        if 'across_channels' not in attrs:
            attrs['across_channels'] = 0

        # update the attributes of the node
        MVNCaffe.update_node_stat(node, attrs)
        return cls.enabled
