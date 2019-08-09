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

import importlib
import logging as log
import mmap
import os
import sys

import numpy as np
from google.protobuf import text_format
from google.protobuf.internal import api_implementation

from mo.graph.graph import Graph
from mo.utils.error import Error, FrameworkError
from mo.utils.utils import refer_to_faq_msg


def import_caffe_pb2(caffe_parser_path: str):
    # import caffe_pb2
    sys.path.insert(0, caffe_parser_path)
    caffe_pb2 = importlib.import_module("caffe_pb2")
    sys.path.pop(0)

    return caffe_pb2


def parse_mean(file_path: str, in_shape: np.ndarray, mean_file_offsets: [tuple, None], caffe_pb2):
    blob = caffe_pb2.BlobProto()
    with open(file_path, 'rb') as file:
        data = file.read()

    if not data:
        raise Error('Mean file "{}" is empty.' + refer_to_faq_msg(5),
                    file_path)

    try:
        blob.ParseFromString(data)
        data = np.array(blob.data)  # pylint: disable=no-member

        if blob.HasField('channels') or blob.HasField('height') or blob.HasField('width'):
            data = data.reshape(blob.channels, blob.height, blob.width)  # pylint: disable=no-member
        else:
            data = data.reshape(blob.shape.dim)  # pylint: disable=no-member
        # crop mean image according to input size
        if in_shape[2] > data.shape[1] or in_shape[3] > data.shape[2]:
            raise Error(
                'Input image of shape {} is larger than mean image {} from file "{}". ' +
                refer_to_faq_msg(4),
                in_shape,
                data.shape,
                file_path
            )

        if mean_file_offsets is not None and len(mean_file_offsets) == 2:
            offset_x = mean_file_offsets[0]
            offset_y = mean_file_offsets[1]
        else:
            offset_x = int((data.shape[1] - in_shape[2]) / 2)
            offset_y = int((data.shape[2] - in_shape[3]) / 2)

        mean = []
        for i in range(in_shape[1]):
            data_channel = np.zeros(in_shape[2] * in_shape[3], dtype=np.float32)
            for x in range(in_shape[2]):
                for y in range(in_shape[3]):
                    data_channel[x * in_shape[3] + y] = data[i, x + offset_x, y + offset_y]
            mean.append(data_channel)

        return mean

    except Exception as err:
        raise Error(
            'While processing mean file "{}": {}. Probably mean file has incorrect format. ' +
            refer_to_faq_msg(6),
            file_path,
            str(err)) from err


def load_caffe_proto_model(caffe_pb2, proto_path: str, model_path: [str, None] = None):
    # 1. python protobuf is used
    if api_implementation._implementation_type == 'python':
        message = 'Please expect that Model Optimizer conversion might be slow. ' \
                  'You are currently using Python protobuf library implementation. \n'
        try:
            from google.protobuf.pyext import cpp_message
            # Check os windows and env variable PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION
            if os.name == 'nt' and os.environ.get('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', default='') != 'cpp':
                # 2. cpp implementaion is available but not used
                message += 'However, cpp implementation is available, you can boost ' \
                           'model conversion by setting PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION env variable to cpp. \n' \
                           'Run: set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp \n'
        except ImportError:
            # 3. cpp implementaion is not available
            message += 'However you can use the C++ protobuf implementation that is supplied with the OpenVINO toolkit' \
                       'or build protobuf library from sources. \n' \
                       'Navigate to "install_prerequisites" folder and run: ' \
                       'python -m easy_install protobuf-3.5.1-py($your_python_version)-win-amd64.egg \n' \
                       'set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp'
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
                  '      2. {} does not have a valid structure, for example, it was downloaded as html\n'.format(proto_path) +
                  '      3. {} contains custom layers or attributes that are not supported\n'.format(proto_path) +
                  '         in Model Optimizer by default.\n\n' +
                  '    After you made sure that {} has a valid structure and still see this issue, then\n'.format(proto_path) +
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
        log.error('Exception message: {}\n\n'.format(e) +
                  '    Possible reasons:\n' +
                  '      1. {} does not exist\n'.format(model_path) +
                  '      2. {} does not have a valid structure\n'.format(model_path), extra={'framework_error': True})
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


def caffe_pb_to_nx(proto, model):
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
    graph = Graph()
    # Blobs in prototxt model can be reused by inplace layer.
    # This requires loading of pb layers in order and tracking the latest
    # layer that writes a particular blob.
    blob_producers = {}  # maps layer blob name to the layer name and port
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
        input_dims = [np.array(list(proto.input_dim), dtype=np.int64)]
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
        input_dims = [np.array(proto.input_shape[0].dim, dtype=np.int64)]
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
            input_dims.append(np.array(proto.input_shape[i].dim, dtype=np.int64))
            input_names.append(proto.input[i])

    for i in range(len(input_names)):
        input_name = input_names[i]
        input_dim = input_dims[i]
        # Input is defined at the top level of proto instead of distinct Input layer
        graph.add_node(input_name, pb=None, model_pb=None, type='GlobalInput', name=input_name, shape=input_dim,
                       kind='op')
        blob_producers[input_name] = (input_name, 0)

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
                input_dims.append(np.array(list(dims), dtype=np.int64))
                input_names.append(layer.name)

        layer.name = graph.unique_id(layer.name)
        graph.add_node(layer.name, pb=layer, model_pb=model_layer, kind='op', type='Parameter')

        # connect inputs based on blob_producers dictionary
        for dst_port, bottom in enumerate(layer.bottom):
            src_layer = blob_producers[bottom][0]
            src_port = blob_producers[bottom][1]
            assert (graph.has_node(src_layer))
            edge_attrs = {
                'out': src_port,
                'in': dst_port,
                'name': bottom,
                'fw_tensor_debug_info': [(src_layer, bottom)],  # debug anchor for a framework tensor name and port
                'in_attrs': ['in', 'name'],
                'out_attrs': ['out', 'name'],
                'data_attrs': ['fw_tensor_debug_info']
            }
            graph.add_edge(src_layer, layer.name, **edge_attrs)

        # update blob producers dictionary by output ports
        for src_port, top in enumerate(layer.top):
            if top in blob_producers:
                log.debug("Detected reuse of blob {} by layer {}".format(top, layer.name))
            blob_producers[top] = (layer.name, src_port)

    if len(input_names) <= 0:
        raise Error('The topology contains no "input" layers. ' +
                    refer_to_faq_msg(79))
    return graph, {name: shape for (name, shape) in zip(input_names, input_dims)}
