# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.lstm_cell import LSTMCell
from openvino.tools.mo.front.caffe.extractors.utils import embed_input
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.kaldi.loader.utils import collect_until_token, collect_until_whitespace, get_uint32
from openvino.tools.mo.front.kaldi.utils import read_binary_matrix, read_binary_vector


class LSTMProjectedStreamsFrontExtractor(FrontExtractorOp):
    op = 'lstmprojectedstreams'
    enabled = True

    @classmethod
    def extract(cls, node):
        clip_value = 50
        pb = node.parameters
        res = collect_until_whitespace(pb)
        if res == b'<CellClip>':
            clip_value = get_uint32(pb.read(4))
        collect_until_token(pb, b'FM')
        gifo_x_weights, gifo_x_weights_shape = read_binary_matrix(pb, False)
        gifo_r_weights, gifo_r_weights_shape = read_binary_matrix(pb)
        gifo_biases = read_binary_vector(pb)
        input_gate_weights = read_binary_vector(pb)
        forget_gate_weights = read_binary_vector(pb)
        output_gate_weights = read_binary_vector(pb)

        projection_weights, projection_weights_shape = read_binary_matrix(pb)

        mapping_rule = {'gifo_x_weights_shape': gifo_x_weights_shape,
                        'gifo_r_weights_shape': gifo_r_weights_shape,
                        'projection_weights_shape': projection_weights_shape,
                        'clip_value': clip_value,
                        'format': 'kaldi',
                        }

        embed_input(mapping_rule, 1, 'gifo_x_weights', gifo_x_weights)
        embed_input(mapping_rule, 2, 'gifo_r_weights', gifo_r_weights)
        embed_input(mapping_rule, 3, 'gifo_biases', gifo_biases)
        embed_input(mapping_rule, 4, 'input_gate_weights', input_gate_weights)
        embed_input(mapping_rule, 5, 'forget_gate_weights', forget_gate_weights)
        embed_input(mapping_rule, 6, 'output_gate_weights', output_gate_weights)
        embed_input(mapping_rule, 7, 'projection_weights', projection_weights)

        LSTMCell.update_node_stat(node, mapping_rule)
        return cls.enabled


class LSTMProjectedFrontExtractor(FrontExtractorOp):
    op = 'lstmprojected'
    enabled = True

    @classmethod
    def extract(cls, node):
        return LSTMProjectedStreamsFrontExtractor.extract(node)
