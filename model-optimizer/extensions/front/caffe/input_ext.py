# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.parameter import Parameter
from mo.front.caffe.extractors.utils import dim_to_shape
from mo.front.extractor import FrontExtractorOp


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
