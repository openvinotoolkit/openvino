# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.reverse_sequence import ReverseSequence
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class ReverseSequenceExtractor(FrontExtractorOp):
    op = 'ReverseSequence'
    enabled = True

    @classmethod
    def extract(cls, node):
        batch_axis = onnx_attr(node, 'batch_axis', 'i', default=1)
        time_axis = onnx_attr(node, 'time_axis', 'i', default=0)

        attrs = {
            'batch_axis': batch_axis,
            'seq_axis': time_axis,
        }
        ReverseSequence.update_node_stat(node, attrs)
        return cls.enabled
