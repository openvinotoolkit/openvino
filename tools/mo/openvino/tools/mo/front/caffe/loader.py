# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import logging as log
import mmap
import os
import sys

import numpy as np
from google.protobuf import text_format
from google.protobuf.internal import api_implementation

from openvino.tools.mo.front.common.partial_infer.utils import mo_array, int64_array
from openvino.tools.mo.front.extractor import add_outputs_identity
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.utils.error import Error, FrameworkError
from openvino.tools.mo.utils.utils import refer_to_faq_msg


def import_caffe_pb2(caffe_parser_path: str):
    # import caffe_pb2
    sys.path.insert(0, caffe_parser_path)
    caffe_pb2 = importlib.import_module("caffe_pb2")
    sys.path.pop(0)

    return caffe_pb2

def load_caffe_proto_model(caffe_pb2, proto_path: str, model_path: [str, None] = None):
    # 1. python protobuf is used
    if api_implementation._implementation_type == 'python':
        message = 'Please expect that Model Optimizer conversion might be slow. ' \
                  'You are currently using Python protobuf library implementation. \n'
        try:
            from google.protobuf.pyext import cpp_message
            # Check os windows and env variable PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION
            if os.name == 'nt' and os.environ.get('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', default='') != 'cpp':
                # 2. cpp implementation is available but not used
                message += 'However, cpp implementation is available, you can boost ' \
                           'model conversion by setting PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION env variable to cpp. \n' \
                           'Run: set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp \n'
        except ImportError:
            # 3. cpp implementation is not available
            message += 'Check that your protobuf package version is aligned with requirements_caffe.txt.'
        print(message + '\n\n' + refer_to_faq_msg(80))

    # Read proto layers
    try:
        proto = caffe_pb2.NetParameter()
        with open(proto_path, "r") as file:
            text_format.Merge(str(file.read()), proto)
    except Exception as e:
        log.error('Exception message: {}\n\n'.format(e) +
                  '    Possible reasons:\n' +
                  '      1. {} does not exist\n'.format(proto_path) +
                  '      2. {} does not have a valid structure, for example, it was downloaded as html\n'.format(
                      proto_path) +
                  '      3. {} contains custom layers or attributes that are not supported\n'.format(proto_path) +
                  '         in Model Optimizer by default.\n\n' +
                  '    After you made sure that {} has a valid structure and still see this issue, then\n'.format(
                      proto_path) +
                  '    you need to generate a python parser for caffe.proto that was used when the model\n' +
                  '    was created.\n' +
                  '    Run "python3 generate_caffe_pb2.py --input_proto ${PATH_TO_CAFFE}/src/caffe/proto/caffe.proto"' +
                  refer_to_faq_msg(1) + '\n\n', extra={'framework_error': True})
        raise FrameworkError('Model Optimizer is not able to parse {}'.format(proto_path)) from e

    # Read model layer if exists
    model = None
    try:
        if model_path:
            model = caffe_pb2.NetParameter()
            with open(model_path, "rb") as infile:
                map = mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ)
                model.MergeFromString(map)
    except Exception as e:
        third_point = ''
        if api_implementation._implementation_type == 'python':
            third_point = '      3. Python protobuf implementation was used. Some models can\'t be converted ' + \
                          ' in this configuration. Please, use Python version with existing cpp implementation of ' + \
                          'protobuf library or build it by yourself\n' + refer_to_faq_msg(103)
        log.error('Exception message: {}\n\n'.format(e) +
                  '    Possible reasons:\n' +
                  '      1. {} does not exist\n'.format(model_path) +
                  '      2. {} does not have a valid structure\n'.format(model_path) + third_point,
                  extra={'framework_error': True})
        raise FrameworkError('Model Optimizer is not able to parse {}'.format(model_path)) from e

    return proto, model


def get_layers(proto):
    if len(proto.layer):
        return proto.layer
    elif len(proto.layers):
        return proto.layers
    else:
        raise Error('Invalid proto file: there is neither "layer" nor "layers" top-level messages. ' +
                    refer_to_faq_msg(7))


def caffe_pb_to_nx(graph, proto, model):
    """
    Converts proto/model layers to a graph. Edges are restored by bottom/top attributes.
    Graph nodes has two attributes: pb for prototxt definition and model_pb for caffemodel definition.

    Parameters
    ----------
    proto : NetParameter
       Protobuf message for NetParameter, representing .prototxt.
    model : NetParameter
       Protobuf message for NetParameter, representing .caffemodel.

    Returns
    ----------
        Graph
        built NX Directed graph.
    """
    # Blobs in prototxt model can be reused by inplace layer.
    # This requires loading of pb layers in order and tracking the latest
    # layer that writes a particular blob.
    blob_producers = {}  # maps layer blob name to node id in graph, port and layer name
    proto_layers = get_layers(proto)
    model_layers = None
    if model:
        model_layers = get_layers(model)

    input_dims = []
    input_names = []
    if len(proto.input_dim) > 0 and len(list(proto.input)) > 1:
        # example of proto input
        # input: "data"
        # input_dim: 1
        # input_dim: 3
        # input_dim: 500
        # input_dim: 500
        # input: "info"
        # input_dim: 1
        # input_dim: 3
        raise Error('Old-style inputs (via "input_dims") are not supported. ' +
                    'Please specify inputs via  "input_shape". ' +
                    refer_to_faq_msg(8))
    elif len(list(proto.input)) == 1 and len(list(proto.input_dim)):
        # example of proto input
        # input: "data"
        # input_dim: 1
        # input_dim: 3
        # input_dim: 500
        # input_dim: 500
        input_dims = [int64_array(list(proto.input_dim))]
        input_names = [proto.input[0]]

    elif len(list(proto.input)) == 1 and len(list(proto.input_shape)):
        # example of proto input
        # input: "data"
        # input_shape
        # {
        #     dim: 1
        #     dim: 3
        #     dim: 227
        #     dim: 227
        # }
        input_dims = [int64_array(proto.input_shape[0].dim)]
        input_names = [proto.input[0]]

    elif len(proto.input_shape) > 0:
        # example of proto input
        # input: "data"
        # input_shape
        # {
        #     dim: 1
        #     dim: 3
        #     dim: 600
        #     dim: 1000
        # }
        # input: "im_info"
        # input_shape
        # {
        #     dim: 1
        #     dim: 3
        # }
        for i in range(len(proto.input_shape)):
            input_dims.append(int64_array(proto.input_shape[i].dim))
            input_names.append(proto.input[i])

    for i in range(len(input_names)):
        input_name = input_names[i]
        input_dim = input_dims[i]
        # Input is defined at the top level of proto instead of distinct Input layer
        graph.add_node(input_name, pb=None, model_pb=None, type='GlobalInput', name=input_name, shape=input_dim,
                       kind='op')
        blob_producers[input_name] = (input_name, 0, input_name)

    used_blobs = set()
    for i, layer in enumerate(proto_layers):

        model_layer = None

        if model_layers:
            for ml in model_layers:
                if ml.name == layer.name:
                    model_layer = ml
                    break
        if layer.type == 'Input':
            if hasattr(layer, 'input_param'):
                input_param = layer.input_param
            else:
                raise Error('Input layer has no input dims. ' +
                            refer_to_faq_msg(8))
            if hasattr(input_param, 'shape'):
                """
                example of proto input
                layer
                {
                    name: "data"
                    type: "Input"
                    top: "data"
                    input_param {shape: {dim: 1 dim: 3 dim: 600 dim: 1000}}
                }

                layer
                {
                    name: "im_info"
                    type: "Input"
                    top: "im_info"
                    input_param {shape: {dim: 1 dim: 3}}
                }
                """
                dims = map(int, list(filter(None, str(list(input_param.shape)[0]).split('dim:'))))
                input_dims.append(int64_array(list(dims)))
                input_names.append(layer.name)

        node_id = graph.unique_id(layer.name)
        graph.add_node(node_id, pb=layer, model_pb=model_layer, kind='op', type='Parameter')
        if hasattr(graph, 'op_names_statistic') and hasattr(layer, 'type'):
            graph.op_names_statistic[layer.type] += 1

        # connect inputs based on blob_producers dictionary
        for dst_port, bottom in enumerate(layer.bottom):
            add_edge_caffe(graph, bottom, node_id, blob_producers, dst_port)
            used_blobs.add(bottom)

        # update blob producers dictionary by output ports
        for src_port, top in enumerate(layer.top):
            if top in blob_producers:
                log.debug("Detected reuse of blob {} by layer {}".format(top, node_id))
            blob_producers[top] = (node_id, src_port, layer.name)

    # Tensor names information corresponding to a node is stored on outgoing edges.
    # As output nodes do not have outgoing edges, fake outputs are required. In the following code
    # for each output Identity node is added, and tensor name for the output is kept
    # on (output, fake output) edge. After Result nodes adding transformation fake outputs
    # are deleted from graph.
    all_blobs = set(blob_producers.keys())
    add_outputs_identity(graph, all_blobs - used_blobs, add_edge_caffe,
                         {'blob_producers': blob_producers, 'dst_port': 0})

    if len(input_names) <= 0:
        raise Error('The topology contains no "input" layers. ' +
                    refer_to_faq_msg(79))
    return {fake_node_name: shape for (fake_node_name, shape) in zip(input_names, input_dims)}


def add_edge_caffe(graph: Graph, bottom: str, dst_layer: str, blob_producers: dict, dst_port: int):
    """
    Creates an edge and adds it to the graph.
    """
    src_layer = blob_producers[bottom][0]
    src_port = blob_producers[bottom][1]
    edge_attrs = {
        'out': src_port,
        'in': dst_port,
        'name': bottom,
        # debug anchor for a framework name and tensor name
        'fw_tensor_debug_info': [(blob_producers[bottom][2], bottom)],
        'in_attrs': ['in', 'name'],
        'out_attrs': ['out', 'name'],
        'data_attrs': ['fw_tensor_debug_info']
    }
    graph.add_edge(src_layer, dst_layer, **edge_attrs)
