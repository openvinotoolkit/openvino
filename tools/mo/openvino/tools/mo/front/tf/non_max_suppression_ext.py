# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.ops.non_max_suppression import NonMaxSuppression
from openvino.tools.mo.front.extractor import FrontExtractorOp


class NonMaxSuppressionV2Extractor(FrontExtractorOp):
    op = 'NonMaxSuppressionV2'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'sort_result_descending': 1, 'box_encoding': 'corner', 'output_type': np.int32}
        NonMaxSuppression.update_node_stat(node, attrs)
        return cls.enabled


class NonMaxSuppressionV3Extractor(FrontExtractorOp):
    op = 'NonMaxSuppressionV3'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'sort_result_descending': 1, 'box_encoding': 'corner', 'output_type': np.int32}
        NonMaxSuppression.update_node_stat(node, attrs)
        return cls.enabled


class NonMaxSuppressionV4Extractor(FrontExtractorOp):
    op = 'NonMaxSuppressionV4'
    enabled = True

    @classmethod
    def extract(cls, node):
        pad_to_max_output_size = node.pb.attr["pad_to_max_output_size:"].b
        if not pad_to_max_output_size:
            log.warning('The attribute "pad_to_max_output_size" of node {} is equal to False which is not supported. '
                        'Forcing it to be equal to True'.format(node.soft_get('name')))
        attrs = {'sort_result_descending': 1, 'box_encoding': 'corner', 'output_type': np.int32}
        NonMaxSuppression.update_node_stat(node, attrs)
        return cls.enabled


class NonMaxSuppressionV5Extractor(FrontExtractorOp):
    op = 'NonMaxSuppressionV5'
    enabled = True

    @classmethod
    def extract(cls, node):
        pad_to_max_output_size = node.pb.attr["pad_to_max_output_size:"].b
        if not pad_to_max_output_size:
            log.warning('The attribute "pad_to_max_output_size" of node {} is equal to False which is not supported. '
                        'Forcing it to be equal to True'.format(node.soft_get('name')))
        attrs = {'sort_result_descending': 1, 'box_encoding': 'corner', 'output_type': np.int32}
        NonMaxSuppression.update_node_stat(node, attrs)
        return cls.enabled
