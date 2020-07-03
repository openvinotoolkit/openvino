"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import logging as log

import numpy as np

from extensions.ops.non_max_suppression import NonMaxSuppression, TFNonMaxSuppressionV5
from mo.front.extractor import FrontExtractorOp


class NonMaxSuppressionV3Extractor(FrontExtractorOp):
    op = 'NonMaxSuppressionV3'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'sort_result_descending': 1, 'center_point_box': 0, 'output_type': np.int32}
        NonMaxSuppression.update_node_stat(node, attrs)
        return cls.enabled


class NonMaxSuppressionV4Extractor(FrontExtractorOp):
    op = 'NonMaxSuppressionV4'
    enabled = True

    @classmethod
    def extract(cls, node):
        pad_to_max_output_size = node.pb.attr["pad_to_max_output_size:"].b
        if not pad_to_max_output_size:
            log.warning('The attribute "pad_to_max_output_size" of node {} is equal to False which is not supported.'
                        'Forcing it to be equal to True'.format(node.soft_get('name')))
        attrs = {'sort_result_descending': 1, 'box_encoding': 'corner', 'output_type': np.int32}
        NonMaxSuppression.update_node_stat(node, attrs)
        return cls.enabled


class NonMaxSuppressionV5Extractor(FrontExtractorOp):
    op = 'NonMaxSuppressionV5'
    enabled = True

    @classmethod
    def extract(cls, node):
        pad_to_max_output_size = node.pb.attr['pad_to_max_output_size:'].b
        if not pad_to_max_output_size:
            log.warning('The attribute "pad_to_max_output_size" of node {} is equal to False which is not supported.'
                        'Forcing it to be equal to True'.format(node.soft_get('name')))
        TFNonMaxSuppressionV5.update_node_stat(node, {'pad_to_max_output_size': pad_to_max_output_size})
        return cls.enabled
