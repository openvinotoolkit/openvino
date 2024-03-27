# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import float32_array, int64_array
from openvino.tools.mo.ops.GRU import GRU
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class GRUFrontExtractor(FrontExtractorOp):
    op = 'GRU'
    enabled = True

    @classmethod
    def extract(cls, node):
        activation_alpha = onnx_attr(node, 'activation_alpha', 'floats',
                                     default=None, dst_type=lambda x: float32_array(x))
        activation_beta = onnx_attr(node, 'activation_beta', 'floats',
                                    default=None, dst_type=lambda x: float32_array(x))
        activations = onnx_attr(node, 'activations', 'strings', default=None,
                                dst_type=lambda x: list(map(lambda s: s.decode(encoding="utf-8").lower(), list(x))))
        clip = onnx_attr(node, 'clip', 'f', default=None)
        linear_before_reset = onnx_attr(node, 'linear_before_reset', 'i', default=0)

        attrs = {
            'batch_dim': 1,
            'sequence_dim': 0,
            'blobs_wrb': True,
            'has_num_directions': True,
            'num_layers': 1,
            'format': 'onnx',
            'multilayers': False,
            'gate_order': [0, 1, 2],

            # ONNX - specific attrs
            'activation_alpha': activation_alpha,
            'activation_beta': activation_beta,
            'activations': activations,
            'clip': clip,
            'direction': onnx_attr(node, 'direction', 's', b'forward').decode().lower(),
            'hidden_size': int64_array(onnx_attr(node, 'hidden_size', 'i')),
            'linear_before_reset': linear_before_reset,
        }

        GRU.update_node_stat(node, attrs)
        return cls.enabled
