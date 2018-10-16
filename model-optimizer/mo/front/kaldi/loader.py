"""
 Copyright (c) 2018 Intel Corporation

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
import os
from io import IOBase

from mo.front.kaldi.extractor import common_kaldi_fields
from mo.front.kaldi.utils import get_uint32, get_uint16, KaldiNode
from mo.graph.graph import unique_id, Node
from mo.utils.error import Error

import networkx as nx
import numpy as np

from mo.utils.utils import refer_to_faq_msg


def read_placeholder(file_desc):
    """
    Placeholder is like: |FW | or |FV | - they take 3 spaces and appear before a matrix or a vector respectively
    :param file_path:
    :return:
    """
    file_desc.read(3)


def read_binary_matrix(file_desc, skip: bool = False):
    if not skip:
        read_placeholder(file_desc)
    rows_number = read_binary_integer_token(file_desc)
    cols_number = read_binary_integer_token(file_desc)
    # to compare: ((float *)a->buffer())[10]
    return read_blob(file_desc, rows_number * cols_number), (rows_number, cols_number)


def read_binary_vector(file_desc):
    read_placeholder(file_desc)
    elements_number = read_binary_integer_token(file_desc)
    return read_blob(file_desc, elements_number)


def collect_until_token(f, token):
    while True:
        # usually there is the following structure <CellDim> DIM<ClipGradient> VALUEFM
        res = collect_until_whitespace(f)
        if res[-2:] == token:
            return


class KaldiLayer:
    def __init__(self, f, graph, layer_i, layer_o, layer_name, prev_layer_name, is_switch_board_style=False,
                 type='General'):
        self.f = f
        self.graph = graph
        self.type = type
        self.layer_i = layer_i
        self.layer_o = layer_o
        self.layer_name = layer_name
        self.prev_layer_name = prev_layer_name
        self.is_switch_board_style = is_switch_board_style
        self.attrs = dict(type=type)
        self.weights = None
        self.biases = None

    def construct_sub_graph(self):
        return add_single_node(self.graph, self.layer_name, self.prev_layer_name, self.attrs, self.weights, self.biases)

    def load_build(self):
        return self.construct_sub_graph()


class SigmoidKaldiLayer(KaldiLayer):
    def load_build(self):
        self.attrs.update({
            'operation': 'sigmoid'
        })
        return self.construct_sub_graph()


class AffineTransformKaldiLayer(KaldiLayer):
    def load_weights_biases_attrs(self):
        collect_until_token(self.f, b'FM')
        self.weights, weights_shape = read_binary_matrix(self.f, skip=True)
        self.biases = read_binary_vector(self.f)
        self.attrs = {
            'num_output': self.layer_o,
            'bias_term': True,
            'weights_shape': weights_shape,
            'type': 'AffineTransform'
        }

    def load_build(self):
        self.load_weights_biases_attrs()
        return self.construct_sub_graph()


class LSTMProjectedStreamsKaldiLayer(KaldiLayer):
    def __init__(self, f, graph, layer_i, layer_o, layer_name, prev_layer_name, is_switch_board_style=False,
                 type='General'):
        super().__init__(f, graph, layer_i, layer_o, layer_name, prev_layer_name, is_switch_board_style, type)
        self.clip_value = None
        self.gifo_x_weights = None
        self.gifo_r_weights = None
        self.gifo_biases = None
        self.input_gate_weights = None
        self.forget_gate_weights = None
        self.output_gate_weights = None
        self.projection_weights = None
        self.gifo_x_weights_shape = None
        self.gifo_r_weights_shape = None
        self.projection_weights_shape = None

    def load_weights_biases_attrs(self):
        self.clip_value = 1 if self.is_switch_board_style else 50

        if not self.is_switch_board_style:
            res = collect_until_whitespace(self.f)  # <CellClip>
            if res == b'<CellClip>':
                self.clip_value = get_uint32(self.f.read(4))

            collect_until_token(self.f, b'FM')

        self.gifo_x_weights, self.gifo_x_weights_shape = read_binary_matrix(self.f, skip=True)
        self.gifo_r_weights, self.gifo_r_weights_shape = read_binary_matrix(self.f)
        self.gifo_biases = read_binary_vector(self.f)
        self.input_gate_weights = read_binary_vector(self.f)
        self.forget_gate_weights = read_binary_vector(self.f)
        self.output_gate_weights = read_binary_vector(self.f)

        if not self.is_switch_board_style:
            self.projection_weights, self.projection_weights_shape = read_binary_matrix(self.f)

    def load_build(self):
        self.load_weights_biases_attrs()
        return self.construct_sub_graph()

    def construct_sub_graph(self):
        self.attrs.update(dict(gifo_x_weights=self.gifo_x_weights, gifo_r_weights=self.gifo_r_weights,
                               gifo_biases=self.gifo_biases, input_gate_weights=self.input_gate_weights,
                               forget_gate_weights=self.forget_gate_weights,
                               clip_value=self.clip_value,
                               output_gate_weights=self.output_gate_weights,
                               projection_weights=self.projection_weights,
                               gifo_x_weights_shape=self.gifo_x_weights_shape,
                               gifo_r_weights_shape=self.gifo_r_weights_shape,
                               projection_weights_shape=self.projection_weights_shape,
                               type='LSTMProjectedStreams'))
        return add_single_node(self.graph, self.layer_name, self.prev_layer_name, self.attrs, self.weights, self.biases)


class ConvolutionKaldiLayer(KaldiLayer):
    def __init__(self, f, graph, layer_i, layer_o, layer_name, prev_layer_name, is_switch_board_style=False):
        super().__init__(f, graph, layer_i, layer_o, layer_name, prev_layer_name, is_switch_board_style, 'Convolution')
        self.kernel = None
        self.stride = None
        self.output = None
        self.weights_shape = None
        self.shape = None
        self.patch_stride = None

    def load_build(self):
        '''
        /* Prepare feature patches, the layout is:
        274      |----------|----------|----------|---------| (in = spliced frames)
        275       xxx        xxx        xxx        xxx        (x = selected elements)
        276
        277        xxx : patch dim
        278         xxx
        279        ^---: patch step
        280      |----------| : patch stride
        281
        282        xxx-xxx-xxx-xxx : filter dim
        283
        '''
        self.kernel = read_token_value(self.f, b'<PatchDim>')
        self.stride = read_token_value(self.f, b'<PatchStep>')
        self.patch_stride = read_token_value(self.f, b'<PatchStride>')

        if (self.patch_stride - self.kernel) % self.stride != 0:
            raise Error(
                'Kernel size and stride does not correspond to `patch_stride` attribute of Convolution layer. ' +
                refer_to_faq_msg(93))

        do_loop = True
        while do_loop:
            self.f.read(1)
            first_char = self.f.read(1)
            self.f.seek(-2, os.SEEK_CUR)
            if first_char == b'L':
                read_token_value(self.f, b'<LearnRateCoef>')
            elif first_char == b'B':
                read_token_value(self.f, b'<BiasLearnRateCoef>')
            elif first_char == b'M':
                read_token_value(self.f, b'<MaxNorm>')
            elif first_char == b'!':
                read_token_value(self.f, b'<EndOfComponent>')
                do_loop = False
            else:
                do_loop = False
        self.load_weights_biases_attrs()

        self.output = self.biases.shape[0]
        if self.weights_shape[0] != self.output:
            raise Error('Weights shape does not correspond to the `output` attribute of Convolution layer. ' +
                        refer_to_faq_msg(93))
        self.attrs.update({
            'kernel': self.kernel,
            'stride': self.stride,
            'output': self.output,
            'bias_term': True,
            'patch_stride': self.patch_stride
        })
        return self.construct_sub_graph()

    def load_weights_biases_attrs(self):
        collect_until_whitespace(self.f)
        self.weights, self.weights_shape = read_binary_matrix(self.f)
        collect_until_whitespace(self.f)
        self.biases = read_binary_vector(self.f)


class PoolingKaldiLayer(KaldiLayer):
    def __init__(self, f, graph, layer_i, layer_o, layer_name, prev_layer_name, is_switch_board_style=False,
                 pool_method='Max'):
        super().__init__(f, graph, layer_i, layer_o, layer_name, prev_layer_name, is_switch_board_style, 'Pooling')
        self.pad = 0
        self.window = None
        self.pool_method = pool_method
        self.stride = None

    def load_build(self):
        self.window = read_token_value(self.f, b'<PoolSize>')
        self.stride = read_token_value(self.f, b'<PoolStep>')
        pool_stride = read_token_value(self.f, b'<PoolStride>')

        self.attrs.update({
            'kernel': self.window,
            'stride': self.stride,
            'pool_stride': pool_stride,
            'pool_method': self.pool_method
        })
        return self.construct_sub_graph()


class ScaleShiftKaldiLayer(KaldiLayer):
    def __init__(self, f, graph, layer_i, layer_o, layer_name, prev_layer_name, is_switch_board_style=False,
                 weights=None):
        super().__init__(f, graph, layer_i, layer_o, layer_name, prev_layer_name, is_switch_board_style, 'ScaleShift')
        self.weights = weights
        self.bias_term = False

    def load_build(self):
        if collect_until_whitespace(self.f) == b'<AddShift>':
            self.layer_o = read_binary_integer_token(self.f)
            self.layer_o = read_binary_integer_token(self.f)
            self.biases = read_binary_vector(self.f)
            self.bias_term = True
            self.attrs.update({'bias_term': self.bias_term})
        return self.construct_sub_graph()


class RescaleKaldiLayer(KaldiLayer):
    def __init__(self, f, graph, layer_i, layer_o, layer_name, prev_layer_name, is_switch_board_style=False):
        super().__init__(f, graph, layer_i, layer_o, layer_name, prev_layer_name, is_switch_board_style, 'ScaleShift')
        self.weights = None
        self.bias_term = False

    def load_build(self):
        if self.f.read(1) == b'<':
            self.f.seek(-1, os.SEEK_CUR)
            read_token_value(self.f, b'<LearnRateCoef>')
        else:
            self.f.seek(-1, os.SEEK_CUR)
        self.weights = read_binary_vector(self.f)
        next_token = collect_until_whitespace(self.f)
        if next_token == b'<!EndOfComponent>':
            next_token = collect_until_whitespace(self.f)
        if next_token == b'<AddShift>':
            read_binary_integer_token(self.f)  # input layer
            self.layer_o = read_binary_integer_token(self.f)
            if self.f.read(1) == b'<':
                self.f.seek(-1, os.SEEK_CUR)
                read_token_value(self.f, b'<LearnRateCoef>')
            else:
                self.f.seek(-1, os.SEEK_CUR)
            self.biases = read_binary_vector(self.f)
            self.bias_term = True
            self.attrs.update({'bias_term': self.bias_term})
        else:
            self.f.seek(-len(next_token), os.SEEK_CUR)
        return self.construct_sub_graph()


class ParallelKaldiLayer(KaldiLayer):
    def __init__(self, f, graph, layer_i, layer_o, layer_name, prev_layer_name, is_switch_board_style=False):
        super().__init__(f, graph, layer_i, layer_o, layer_name, prev_layer_name, is_switch_board_style, 'Parallel')
        self.output_nodes = []
        self.edge_attrs = {
            'out': None,
            'in': 0,
            'name': None,
            'fw_tensor_name': None,
            'fw_tensor_port': None,
            'in_attrs': ['in', 'name'],
            'out_attrs': ['out', 'name'],
            'data_attrs': ['fw_tensor_name', 'fw_tensor_port']
        }

    def load_build(self):
        nnet_count = read_token_value(self.f, b'<NestedNnetCount>')
        slice_id = add_single_node(self.graph, 'Slice', self.prev_layer_name,
                                   {'type': 'Slice', 'axis': 1, 'slice_point': []}, None, None)
        for i in range(nnet_count):
            read_token_value(self.f, b'<NestedNnet>')
            graph, shape = load_kaldi_nnet_model(self.f, None)
            input_nodes = [n for n in graph.nodes(data=True) if n[1]['type'] == 'GlobalInput']
            for input_node in input_nodes:
                shape_subgraph = input_node[1]['shape']
                if i != nnet_count - 1:
                    self.graph.node[slice_id]['pb'].slice_point.append(shape_subgraph[1])
                graph.remove_node(input_node[0])
            mapping = {node: unique_id(self.graph, node) for node in graph.nodes(data=False) if node in self.graph}
            g = nx.relabel_nodes(graph, mapping)
            for val in mapping.values():
                g.node[val]['name'] = val
            self.graph.add_nodes_from(g.nodes(data=True))
            self.graph.add_edges_from(g.edges(data=True))
            sorted_nodes = tuple(nx.topological_sort(g))
            self.edge_attrs['out'] = i
            self.edge_attrs['name'] = sorted_nodes[0]
            self.edge_attrs['fw_tensor_name'] = slice_id
            self.edge_attrs['fw_tensor_port'] = sorted_nodes[0]
            self.graph.add_edge(slice_id, sorted_nodes[0], **self.edge_attrs)
            self.output_nodes.append(sorted_nodes[-1])
        end_token = collect_until_whitespace(self.f)
        if end_token != b'</ParallelComponent>':
            raise Error('Expected token `</ParallelComponent>`, has {}'.format(end_token) + refer_to_faq_msg(99))
        return self.construct_sub_graph()

    def construct_sub_graph(self):
        new_id = unique_id(self.graph, '{}_'.format('Concat'))
        layer = KaldiNode(new_id)
        layer.set_attrs(dict(axis=1))
        layer.type = 'Concat'
        self.graph.add_node(new_id, pb=layer, kind='op')
        self.graph.node[layer.name].update(common_kaldi_fields(Node(self.graph, layer.name)))
        self.edge_attrs['out'] = 0
        self.edge_attrs['name'] = layer.name
        self.edge_attrs['fw_tensor_port'] = layer.name
        for i, output_node in enumerate(self.output_nodes):
            self.edge_attrs['fw_tensor_name'] = output_node
            self.edge_attrs['in'] = i
            self.graph.add_edge(output_node, layer.name, **self.edge_attrs)
        return new_id


def read_token_value(file, token: bytes = b'', value_type: type = np.uint32):
    getters = {
        np.uint32: read_binary_integer_token
    }
    current_token = collect_until_whitespace(file)
    if token != b'' and token != current_token:
        raise Error('Can not load token {} from Kaldi model'.format(token) +
                    refer_to_faq_msg(94))
    return getters[value_type](file)


def read_binary_integer_token(file_path):
    buffer_size = file_path.read(1)
    return get_uint32(file_path.read(buffer_size[0]))


def collect_until_whitespace(file_path):
    res = b''
    while True:
        new_sym = file_path.read(1)
        if new_sym == b' ':
            break
        res += new_sym
    return res


def read_blob(file_path, size):
    float_size = 4
    data = file_path.read(size * float_size)
    return np.fromstring(data, dtype='<f4')


layer_weights_biases_attrs_getter = {
    'affinetransform': AffineTransformKaldiLayer,
    'sigmoid': lambda f, g, i, o, name, prev, style: KaldiLayer(f, g, i, o, name, prev, style, type='Sigmoid'),
    'softmax': lambda f, g, i, o, name, prev, style: KaldiLayer(f, g, i, o, name, prev, style, type='SoftMax'),
    'lstmprojectedstreams': LSTMProjectedStreamsKaldiLayer,
    'lstmprojected': LSTMProjectedStreamsKaldiLayer,
    'maxpoolingcomponent': PoolingKaldiLayer,
    'convolutionalcomponent': ConvolutionKaldiLayer,
    'rescale': RescaleKaldiLayer,
    'parallelcomponent': ParallelKaldiLayer,
}


def add_single_node(graph, layer_name, prev_layer_name, attrs, weights, biases):
    new_id = unique_id(graph, '{}_'.format(layer_name))

    layer = KaldiNode(new_id)
    layer.set_weight(weights)
    layer.set_bias(biases)
    if attrs:
        layer.set_attrs(attrs)

    graph.add_node(layer.name, pb=layer, kind='op')
    graph.node[layer.name].update(common_kaldi_fields(Node(graph, layer.name)))

    edge_attrs = {
        'out': 0,
        'in': 0,
        'name': layer.name,
        'fw_tensor_debug_info': [(prev_layer_name, layer.name)],  # debug anchor for a framework tensor name and port
        'in_attrs': ['in', 'name'],
        'out_attrs': ['out', 'name'],
        'data_attrs': ['fw_tensor_debug_info']
    }

    graph.add_edge(prev_layer_name, layer.name, **edge_attrs)

    return new_id


def find_first_tag(file):
    tag = b''
    while True:
        symbol = file.read(1)
        if tag == b'' and symbol != b'<':
            continue
        tag += symbol
        if symbol != b'>':
            continue
        return tag


def find_first_component(file):
    while True:
        tag = find_first_tag(file)
        component_name = tag.decode('ascii').lower()
        if component_name[1:-1] in layer_weights_biases_attrs_getter.keys() or tag == b'</Nnet>' or tag == b'<EndOfComponent>':
            file.read(1)  # read ' '
            return component_name


def load_kaldi_nnet_model(nnet_path, check_sum: int = 16896):
    """
    Structure of the file is the following:
    magic-number(16896)<Nnet> <Next Layer Name> weights etc.
    :param nnet_path:
    :param check_sum:
    :return:
    """
    if isinstance(nnet_path, str):
        file = open(nnet_path, "rb")
    elif isinstance(nnet_path, IOBase):
        file = nnet_path

    # 1. check the file
    # first element is 16896<Nnet>
    if check_sum and get_uint16(file.read(2)) != check_sum:
        raise Error('File {} does not appear to be a Kaldi file (magic number does not match). ', nnet_path,
                    refer_to_faq_msg(89)
                    )

    while True:
        name = find_first_tag(file)
        if name == b'<Nnet>':
            file.read(1)
            break
        elif len(name) == 6:
            raise Error('Kaldi model should start with <Nnet> tag. ',
                        refer_to_faq_msg(89))
    graph = nx.MultiDiGraph()
    input_name = 'Input'
    graph.add_node(input_name, pb=None, type='GlobalInput', name=input_name, shape=None, kind='op')

    prev_layer_name = input_name
    input_shapes = {}

    while True:
        """
        Typical structure of the layer
        <Layer> |Size of output value in bits|Actual value of output|Size of input value in bits|Actual value of input|\
        FM Matrix|FV Vector| </Layer>
        """
        layer_name = find_first_component(file)
        if layer_name == '</nnet>':
            break
        elif layer_name == '<!endofcomponent>':
            continue
        extracted_name = layer_name[1:-1]

        layer_o = read_binary_integer_token(file)
        layer_i = read_binary_integer_token(file)

        if prev_layer_name == 'Input':
            graph.node['Input']['shape'] = np.array([1, layer_i], dtype=np.int64)

        cls = layer_weights_biases_attrs_getter[extracted_name]
        cls_instance = cls(file, graph, layer_i, layer_o, extracted_name, prev_layer_name, False)

        prev_layer_name = cls_instance.load_build()
    return graph, input_shapes


def read_counts_file(file_path):
    with open(file_path, 'r') as f:
        file_content = f.readlines()
    if len(file_content) > 1:
        raise Error('Expect counts file to be one-line file. ' +
                    refer_to_faq_msg(90))

    counts_line = file_content[0].strip().replace('[', '').replace(']', '')
    try:
        counts = np.fromstring(counts_line, dtype=int, sep=' ')
    except TypeError:
        raise Error('Expect counts file to contain list of integers.' +
                    refer_to_faq_msg(90))
    cutoff = 1.00000001e-10
    counts = [cutoff if count < cutoff else count for count in counts]
    scale = 1.0 / np.sum(counts)
    for idx, count in enumerate(counts):
        val = np.log(scale * count)
        if count == cutoff:
            val += np.iinfo(np.float32).max / 2
        counts[idx] = val
    return counts
