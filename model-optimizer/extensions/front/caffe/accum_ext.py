# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.accum import AccumOp
from mo.front.caffe.collect_attributes import collect_attributes
from mo.front.extractor import FrontExtractorOp


class AccumFrontExtractor(FrontExtractorOp):
    op = 'Accum'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.accum_param

        attrs = collect_attributes(param)
        # update the attributes of the node
        AccumOp.update_node_stat(node, attrs)
        return cls.enabled
