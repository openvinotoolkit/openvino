# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.activation_ops import Elu
from mo.front.caffe.collect_attributes import collect_attributes
from mo.front.extractor import FrontExtractorOp


class ELUFrontExtractor(FrontExtractorOp):
    op = 'ELU'
    enabled = True

    @classmethod
    def extract(cls, node):
        param = node.pb.elu_param
        attrs = collect_attributes(param)

        Elu.update_node_stat(node, attrs)
        return cls.enabled
