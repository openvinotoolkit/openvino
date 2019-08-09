"""
 Copyright (c) 2018-2019 Intel Corporation

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
from extensions.ops.lstm_cell import LSTMCell
from mo.front.caffe.extractors.utils import embed_input
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.loader.utils import collect_until_token, collect_until_whitespace, get_uint32
from mo.front.kaldi.utils import read_binary_matrix, read_binary_vector


class LSTMProjectedStreamsFrontExtractor(FrontExtractorOp):
    op = 'lstmprojectedstreams'
    enabled = True

    @staticmethod
    def extract(node):
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
        return __class__.enabled


class LSTMProjectedFrontExtractor(FrontExtractorOp):
    op = 'lstmprojected'
    enabled = True

    @staticmethod
    def extract(node):
        return LSTMProjectedStreamsFrontExtractor.extract(node)
