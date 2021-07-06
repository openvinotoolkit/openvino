# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from extensions.ops.BatchNormInference import BatchNormInference
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr



class BatchNormalizationExtractor(FrontExtractorOp):
    op = 'BatchNormalization'
    enabled = True

    @classmethod
    def extract(cls, node):
        attr_dict = {
           'data_format': 'NCHW',
           'eps': onnx_attr(node, 'epsilon', 'f', 1e-5),
        }
        BatchNormInference.update_node_stat(node, attr_dict)
        return cls.enabled
