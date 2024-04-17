# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.front.caffe.extractors.utils import dim_to_shape
from openvino.tools.mo.front.extractor import FrontExtractorOp


class InputFrontExtractor(FrontExtractorOp):
    op = 'input'
    enabled = True

    @classmethod
    def extract(cls, node):
        Parameter.update_node_stat(node, {'shape': dim_to_shape(node.pb.input_param.shape[0].dim)})
        return cls.enabled


class GlobalInputFrontExtractor(FrontExtractorOp):
    op = 'globalinput'
    enabled = True

    @classmethod
    def extract(cls, node):
        Parameter.update_node_stat(node)
        return cls.enabled
