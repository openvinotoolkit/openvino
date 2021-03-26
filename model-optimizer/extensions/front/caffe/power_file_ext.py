# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.power_file import PowerFileOp
from mo.front.caffe.collect_attributes import collect_attributes
from mo.front.extractor import FrontExtractorOp


class PowerFileFrontExtractor(FrontExtractorOp):
    op = 'PowerFile'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.power_file_param

        attrs = collect_attributes(param)

        # update the attributes of the node
        PowerFileOp.update_node_stat(node, attrs)
        return cls.enabled
