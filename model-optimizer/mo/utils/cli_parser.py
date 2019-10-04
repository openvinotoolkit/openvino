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

import argparse
import logging as log
import os
import re
import sys
from collections import OrderedDict
from itertools import zip_longest

import numpy as np

from mo.front.extractor import split_node_in_port
from mo.utils import import_extensions
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class DeprecatedStoreTrue(argparse.Action):
    def __init__(self, nargs=0, **kw):
        super().__init__(nargs=nargs, **kw)

    def __call__(self, parser, namespace, values, option_string=None):
        dep_msg = "Use of deprecated cli option {} detected. Option use in the following releases will be fatal. ".format(option_string)
        if 'fusing' in option_string:
            dep_msg += 'Please use --finegrain_fusing cli option instead'
        log.error(dep_msg, extra={'is_warning': True})
        setattr(namespace, self.dest, True)


class CanonicalizePathAction(argparse.Action):
    """
    Expand user home directory paths and convert relative-paths to absolute.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            list_of_values = list()
            if isinstance(values, str):
                if values != "":
                    list_of_values = values.split(',')
            elif isinstance(values, list):
                list_of_values = values
            else:
                raise Error('Unsupported type of command line parameter "{}" value'.format(self.dest))
            list_of_values = [get_absolute_path(path) for path in list_of_values]
            setattr(namespace, self.dest, ','.join(list_of_values))


class CanonicalizePathCheckExistenceAction(CanonicalizePathAction):
    """
    Expand user home directory paths and convert relative-paths to absolute and check specified file or directory
    existence.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, values, option_string)
        names = getattr(namespace, self.dest)
        for name in names.split(','):
            if name != "" and not os.path.exists(name):
                raise Error('The value for command line parameter "{}" must be existing file/directory, '
                            ' but "{}" does not exist.'.format(self.dest, name))


def readable_file(path: str):
    """
    Check that specified path is a readable file.
    :param path: path to check
    :return: path if the file is readable
    """
    if not os.path.isfile(path):
        raise Error('The "{}" is not existing file'.format(path))
    elif not os.access(path, os.R_OK):
        raise Error('The "{}" is not readable'.format(path))
    else:
        return path


def readable_dirs(paths: str):
    """
    Checks that comma separated list of paths are readable directories.
    :param paths: comma separated list of paths.
    :return: comma separated list of paths.
    """
    paths_list = [readable_dir(path) for path in paths.split(',')]
    return ','.join(paths_list)


def readable_dirs_or_empty(paths: str):
    """
    Checks that comma separated list of paths are readable directories of if it is empty.
    :param paths: comma separated list of paths.
    :return: comma separated list of paths.
    """
    if paths:
        return readable_dirs(paths)
    return paths


def readable_dir(path: str):
    """
    Check that specified path is a readable directory.
    :param path: path to check
    :return: path if the directory is readable
    """
    if not os.path.isdir(path):
        raise Error('The "{}" is not existing directory'.format(path))
    elif not os.access(path, os.R_OK):
        raise Error('The "{}" is not readable'.format(path))
    else:
        return path


def writable_dir(path: str):
    """
    Checks that specified directory is writable. The directory may not exist but it's parent or grandparent must exist.
    :param path: path to check that it is writable.
    :return: path if it is writable
    """
    if path is None:
        raise Error('The directory parameter is None')
    if os.path.exists(path):
        if os.path.isdir(path):
            if os.access(path, os.W_OK):
                return path
            else:
                raise Error('The directory "{}" is not writable'.format(path))
        else:
            raise Error('The "{}" is not a directory'.format(path))
    else:
        cur_path = path
        while os.path.dirname(cur_path) != cur_path:
            if os.path.exists(cur_path):
                break
            cur_path = os.path.dirname(cur_path)
        if cur_path == '':
            cur_path = os.path.curdir
        if os.access(cur_path, os.W_OK):
            return path
        else:
            raise Error('The directory "{}" is not writable'.format(cur_path))


def get_common_cli_parser(parser: argparse.ArgumentParser = None):
    if not parser:
        parser = argparse.ArgumentParser()
    common_group = parser.add_argument_group('Framework-agnostic parameters')
    # Common parameters
    common_group.add_argument('--input_model', '-w', '-m',
                              help='Tensorflow*: a file with a pre-trained model ' +
                                   ' (binary or text .pb file after freezing).\n' +
                                   ' Caffe*: a model proto file with model weights',
                              action=CanonicalizePathCheckExistenceAction,
                              type=readable_file)
    common_group.add_argument('--model_name', '-n',
                              help='Model_name parameter passed to the final create_ir transform. ' +
                                   'This parameter is used to name ' +
                                   'a network in a generated IR and output .xml/.bin files.')
    common_group.add_argument('--output_dir', '-o',
                              help='Directory that stores the generated IR. ' +
                                   'By default, it is the directory from where the Model Optimizer is launched.',
                              default=get_absolute_path('.'),
                              action=CanonicalizePathAction,
                              type=writable_dir)
    common_group.add_argument('--input_shape',
                              help='Input shape(s) that should be fed to an input node(s) of the model. '
                                   'Shape is defined as a comma-separated list of integer numbers enclosed in '
                                   'parentheses or square brackets, for example [1,3,227,227] or (1,227,227,3), where '
                                   'the order of dimensions depends on the framework input layout of the model. '
                                   'For example, [N,C,H,W] is used for Caffe* models and [N,H,W,C] for TensorFlow* '
                                   'models. Model Optimizer performs necessary transformations to convert the shape to '
                                   'the layout required by Inference Engine (N,C,H,W). The shape should not contain '
                                   'undefined dimensions (? or -1) and should fit the dimensions defined in the input '
                                   'operation of the graph. If there are multiple inputs in the model, --input_shape '
                                   'should contain definition of shape for each input separated by a comma, for '
                                   'example: [1,3,227,227],[2,4] for a model with two inputs with 4D and 2D shapes. '
                                   'Alternatively, you can specify shapes with the --input option.')
    common_group.add_argument('--scale', '-s',
                              type=float,
                              help='All input values coming from original network inputs will be ' +
                                   'divided by this ' +
                                   'value. When a list of inputs is overridden by the --input ' +
                                   'parameter, this scale ' +
                                   'is not applied for any input that does not match with ' +
                                   'the original input of the model.')
    common_group.add_argument('--reverse_input_channels',
                              help='Switch the input channels order from RGB to BGR (or vice versa). Applied to '
                                   'original inputs of the model if and only if a number of channels equals 3. Applied '
                                   'after application of --mean_values and --scale_values options, so numbers in '
                                   '--mean_values and --scale_values go in the order of channels used in the original '
                                   'model.',
                              action='store_true')
    common_group.add_argument('--log_level',
                              help='Logger level',
                              choices=['CRITICAL', 'ERROR', 'WARN', 'WARNING', 'INFO',
                                       'DEBUG', 'NOTSET'],
                              default='ERROR')
    common_group.add_argument('--input',
                              help='Quoted list of comma-separated input nodes names with shapes ' +
                                   'and values for freezing. The shape and value are specified as space-separated lists. '+
                                   'For example, use the following format to set input port 0 ' +
                                   'of the node `node_name1` with the shape [3 4] as an input node and ' +
                                   'freeze output port 1 of the node `node_name2` with the value [20 15] ' +
                                   'and the shape [2]: ' +
                                   '"0:node_name1[3 4],node_name2:1[2]->[20 15]".')
    common_group.add_argument('--output',
                              help='The name of the output operation of the model. ' +
                                   'For TensorFlow*, do not add :0 to this name.')
    common_group.add_argument('--mean_values', '-ms',
                              help='Mean values to be used for the input image per channel. ' +
                                   'Values to be provided in the (R,G,B) or [R,G,B] format. ' +
                                   'Can be defined for desired input of the model, for example: ' +
                                   '"--mean_values data[255,255,255],info[255,255,255]". ' +
                                   'The exact meaning and order ' +
                                   'of channels depend on how the original model was trained.',
                              default=())
    common_group.add_argument('--scale_values',
                              help='Scale values to be used for the input image per channel. ' +
                                   'Values are provided in the (R,G,B) or [R,G,B] format. ' +
                                   'Can be defined for desired input of the model, for example: ' +
                                   '"--scale_values data[255,255,255],info[255,255,255]". ' +
                                   'The exact meaning and order ' +
                                   'of channels depend on how the original model was trained.',
                              default=())
    # TODO: isn't it a weights precision type
    common_group.add_argument('--data_type',
                              help='Data type for all intermediate tensors and weights. ' +
                                   'If original model is in FP32 and --data_type=FP16 is specified, all model weights ' +
                                   'and biases are quantized to FP16.',
                              choices=["FP16", "FP32", "half", "float"],
                              default='float')
    common_group.add_argument('--disable_fusing',
                              help='Turn off fusing of linear operations to Convolution',
                              action=DeprecatedStoreTrue)
    common_group.add_argument('--disable_resnet_optimization',
                              help='Turn off resnet optimization',
                              action='store_true')
    common_group.add_argument('--finegrain_fusing',
                              help='Regex for layers/operations that won\'t be fused. ' +
                                   'Example: --finegrain_fusing Convolution1,.*Scale.*')
    common_group.add_argument('--disable_gfusing',
                              help='Turn off fusing of grouped convolutions',
                              action=DeprecatedStoreTrue)
    common_group.add_argument('--enable_concat_optimization',
                              help='Turn on concat optimization',
                              action='store_true')
    common_group.add_argument('--move_to_preprocess',
                              help='Move mean values to IR preprocess section',
                              action='store_true')
    # we use CanonicalizeDirCheckExistenceAction instead of readable_dirs to handle empty strings
    common_group.add_argument("--extensions",
                              help="Directory or a comma separated list of directories with extensions. To disable all "
                                   "extensions including those that are placed at the default location, pass an empty "
                                   "string.",
                              default=import_extensions.default_path(),
                              action=CanonicalizePathCheckExistenceAction,
                              type=readable_dirs_or_empty)
    common_group.add_argument("--batch", "-b",
                              type=check_positive,
                              default=None,
                              help="Input batch size")
    common_group.add_argument("--version",
                              action='store_true',
                              help="Version of Model Optimizer")

    common_group.add_argument('--silent',
                              help='Prevent any output messages except those that correspond to log level equals '
                                   'ERROR, that can be set with the following option: --log_level. '
                                   'By default, log level is already ERROR. ',
                              action='store_true',
                              default=False)
    common_group.add_argument('--freeze_placeholder_with_value',
                              help='Replaces input layer with constant node with '
                                   'provided value, for example: "node_name->True". '
                                   'It will be DEPRECATED in future releases. '
                                   'Use --input option to specify a value for freezing.',
                              default=None)
    common_group.add_argument('--generate_deprecated_IR_V2',
                              help='Force to generate legacy/deprecated IR V2 to work with previous versions of the'
                                   ' Inference Engine. The resulting IR may or may not be correctly loaded by'
                                   ' Inference Engine API (including the most recent and old versions of Inference'
                                   ' Engine) and provided as a partially-validated backup option for specific'
                                   ' deployment scenarios. Use it at your own discretion. By default, without this'
                                   ' option, the Model Optimizer generates IR V3.',
                              action=DeprecatedStoreTrue)
    common_group.add_argument('--keep_shape_ops',
                              help='[ Experimental feature ] Enables `Shape` operation with all children keeping. '
                                   'This feature makes model reshapable in Inference Engine',
                              action='store_true', default=False)
    common_group.add_argument('--steps',
                              help='Enables model conversion steps display',
                              action='store_true', default=False)

    return parser


def get_common_cli_options(model_name):
    d = OrderedDict()
    d['input_model'] = '- Path to the Input Model'
    d['output_dir'] = ['- Path for generated IR', lambda x: x if x != '.' else os.getcwd()]
    d['model_name'] = ['- IR output name', lambda x: x if x else model_name]
    d['log_level'] = '- Log level'
    d['batch'] = ['- Batch', lambda x: x if x else 'Not specified, inherited from the model']
    d['input'] = ['- Input layers', lambda x: x if x else 'Not specified, inherited from the model']
    d['output'] = ['- Output layers', lambda x: x if x else 'Not specified, inherited from the model']
    d['input_shape'] = ['- Input shapes', lambda x: x if x else 'Not specified, inherited from the model']
    d['mean_values'] = ['- Mean values', lambda x: x if x else 'Not specified']
    d['scale_values'] = ['- Scale values', lambda x: x if x else 'Not specified']
    d['scale'] = ['- Scale factor', lambda x: x if x else 'Not specified']
    d['data_type'] = ['- Precision of IR', lambda x: 'FP32' if x == 'float' else 'FP16' if x == 'half' else x]
    d['disable_fusing'] = ['- Enable fusing', lambda x: not x]
    d['disable_gfusing'] = ['- Enable grouped convolutions fusing', lambda x: not x]
    d['move_to_preprocess'] = '- Move mean values to preprocess section'
    d['reverse_input_channels'] = '- Reverse input channels'
    return d


def get_caffe_cli_options():
    d = {
        'input_proto': ['- Path to the Input prototxt', lambda x: x],
        'caffe_parser_path': ['- Path to Python Caffe* parser generated from caffe.proto', lambda x: x],
        'mean_file': ['- Path to a mean file', lambda x: x if x else 'Not specified'],
        'mean_file_offsets': ['- Offsets for a mean file', lambda x: x if x else 'Not specified'],
        'k': '- Path to CustomLayersMapping.xml',
        'disable_resnet_optimization': ['- Enable resnet optimization', lambda x: not x],
    }

    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def get_tf_cli_options():
    d = {
        'input_model_is_text': '- Input model in text protobuf format',
        'tensorflow_subgraph_patterns': '- Patterns to offload',
        'tensorflow_operation_patterns': '- Operations to offload',
        'tensorflow_custom_operations_config_update': '- Update the configuration file with input/output node names',
        'tensorflow_use_custom_operations_config': '- Use the config file',
        'tensorflow_object_detection_api_pipeline_config': '- Use configuration file used to generate the model with '
                                                           'Object Detection API',
        'tensorflow_custom_layer_libraries': '- List of shared libraries with TensorFlow custom layers implementation',
        'tensorboard_logdir': '- Path to model dump for TensorBoard'
    }

    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def get_mxnet_cli_options():
    d = {
        'input_symbol': '- Deploy-ready symbol file',
        'nd_prefix_name': '- Prefix name for args.nd and argx.nd files',
        'pretrained_model_name': '- Pretrained model to be merged with the .nd files',
        'save_params_from_nd': '- Enable saving built parameters file from .nd files',
        'legacy_mxnet_model': '- Enable MXNet loader for models trained with MXNet version lower than 1.0.0'
    }

    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def get_kaldi_cli_options():
    d = {
        'counts': '- A file name with full path to the counts file',
        'remove_output_softmax': '- Removes the SoftMax layer that is the output layer',
        'remove_memory': '- Removes the Memory layer and use additional inputs and outputs instead'
    }

    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def get_onnx_cli_options():
    d = {
    }

    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def get_caffe_cli_parser(parser: argparse.ArgumentParser = None):
    """
    Specifies cli arguments for Model Optimizer for Caffe*

    Returns
    -------
        ArgumentParser instance
    """
    if not parser:
        parser = argparse.ArgumentParser()
        get_common_cli_parser(parser=parser)

    caffe_group = parser.add_argument_group('Caffe*-specific parameters')

    caffe_group.add_argument('--input_proto', '-d',
                             help='Deploy-ready prototxt file that contains a topology structure ' +
                                  'and layer attributes',
                             type=str,
                             action=CanonicalizePathCheckExistenceAction)
    caffe_group.add_argument('--caffe_parser_path',
                             help='Path to Python Caffe* parser generated from caffe.proto',
                             type=str,
                             default=os.path.join(os.path.dirname(sys.argv[0]), 'mo', 'front', 'caffe', 'proto'),
                             action=CanonicalizePathCheckExistenceAction)
    caffe_group.add_argument('-k',
                             help='Path to CustomLayersMapping.xml to register custom layers',
                             type=str,
                             default=os.path.join(os.path.dirname(sys.argv[0]), 'extensions', 'front', 'caffe',
                                                  'CustomLayersMapping.xml'),
                             action=CanonicalizePathCheckExistenceAction)
    caffe_group.add_argument('--mean_file', '-mf',
                             help='Mean image to be used for the input. Should be a binaryproto file',
                             default=None,
                             action=CanonicalizePathCheckExistenceAction)
    caffe_group.add_argument('--mean_file_offsets', '-mo',
                             help='Mean image offsets to be used for the input binaryproto file. ' +
                                  'When the mean image is bigger than the expected input, it is cropped. By default, centers ' +
                                  'of the input image and the mean image are the same and the mean image is cropped by ' +
                                  'dimensions of the input image. The format to pass this option is the following: "-mo (x,y)". In this ' +
                                  'case, the mean file is cropped by dimensions of the input image with offset (x,y) ' +
                                  'from the upper left corner of the mean image',
                             default=None)
    caffe_group.add_argument('--disable_omitting_optional',
                             help='Disable omitting optional attributes to be used for custom layers. ' +
                                  'Use this option if you want to transfer all attributes of a custom layer to IR. ' +
                                  'Default behavior is to transfer the attributes with default values and the attributes defined by the user to IR.',
                             action='store_true',
                             default=False)
    caffe_group.add_argument('--enable_flattening_nested_params',
                             help='Enable flattening optional params to be used for custom layers. ' +
                                  'Use this option if you want to transfer attributes of a custom layer to IR with flattened nested parameters. ' +
                                  'Default behavior is to transfer the attributes without flattening nested parameters.',
                             action='store_true',
                             default=False)
    return parser


def get_tf_cli_parser(parser: argparse.ArgumentParser = None):
    """
    Specifies cli arguments for Model Optimizer for TF

    Returns
    -------
        ArgumentParser instance
    """
    if not parser:
        parser = argparse.ArgumentParser()
        get_common_cli_parser(parser=parser)

    tf_group = parser.add_argument_group('TensorFlow*-specific parameters')
    tf_group.add_argument('--input_model_is_text',
                          help='TensorFlow*: treat the input model file as a text protobuf format. If not specified, ' +
                               'the Model Optimizer treats it as a binary file by default.',
                          action='store_true')
    tf_group.add_argument('--input_checkpoint', type=str, default=None, help="TensorFlow*: variables file to load.",
                          action=CanonicalizePathCheckExistenceAction)
    tf_group.add_argument('--input_meta_graph',
                          help='Tensorflow*: a file with a meta-graph of the model before freezing',
                          action=CanonicalizePathCheckExistenceAction,
                          type=readable_file)
    tf_group.add_argument('--saved_model_dir', default=None,
                          help="TensorFlow*: directory representing non frozen model",
                          action=CanonicalizePathCheckExistenceAction,
                          type=readable_dirs)
    tf_group.add_argument('--saved_model_tags', type=str, default=None,
                          help="Group of tag(s) of the MetaGraphDef to load, in string format, separated by ','. "
                               "For tag-set contains multiple tags, all tags must be passed in.")
    tf_group.add_argument('--tensorflow_subgraph_patterns',
                          help='TensorFlow*: a list of comma separated patterns that will be applied to ' +
                               'TensorFlow* node names to ' +
                               'infer a part of the graph using TensorFlow*.')
    tf_group.add_argument('--tensorflow_operation_patterns',
                          help='TensorFlow*: a list of comma separated patterns that will be applied to ' +
                               'TensorFlow* node type (ops) ' +
                               'to infer these operations using TensorFlow*.')
    tf_group.add_argument('--tensorflow_custom_operations_config_update',
                          help='TensorFlow*: update the configuration file with node name patterns with input/output '
                               'nodes information.',
                          action=CanonicalizePathCheckExistenceAction)
    tf_group.add_argument('--tensorflow_use_custom_operations_config',
                          help='TensorFlow*: use the configuration file with custom operation description.',
                          action=CanonicalizePathCheckExistenceAction)
    tf_group.add_argument('--tensorflow_object_detection_api_pipeline_config',
                          help='TensorFlow*: path to the pipeline configuration file used to generate model created '
                               'with help of Object Detection API.',
                          action=CanonicalizePathCheckExistenceAction)
    tf_group.add_argument('--tensorboard_logdir',
                          help='TensorFlow*: dump the input graph to a given directory that should be used with TensorBoard.',
                          default=None,
                          action=CanonicalizePathCheckExistenceAction)
    tf_group.add_argument('--tensorflow_custom_layer_libraries',
                          help='TensorFlow*: comma separated list of shared libraries with TensorFlow* custom '
                               'operations implementation.',
                          default=None,
                          action=CanonicalizePathCheckExistenceAction)
    tf_group.add_argument('--disable_nhwc_to_nchw',
                          help='Disables default translation from NHWC to NCHW',
                          action='store_true')
    return parser


def get_mxnet_cli_parser(parser: argparse.ArgumentParser = None):
    """
    Specifies cli arguments for Model Optimizer for MXNet*

    Returns
    -------
        ArgumentParser instance
    """
    if not parser:
        parser = argparse.ArgumentParser()
        get_common_cli_parser(parser=parser)

    mx_group = parser.add_argument_group('Mxnet-specific parameters')

    mx_group.add_argument('--input_symbol',
                          help='Symbol file (for example, model-symbol.json) that contains a topology structure ' +
                               'and layer attributes',
                          type=str,
                          action=CanonicalizePathCheckExistenceAction)
    mx_group.add_argument("--nd_prefix_name",
                          help="Prefix name for args.nd and argx.nd files.",
                          default=None)
    mx_group.add_argument("--pretrained_model_name",
                          help="Name of a pretrained MXNet model without extension and epoch number. This model will be merged with args.nd and argx.nd files",
                          default=None)
    mx_group.add_argument("--save_params_from_nd",
                          action='store_true',
                          help="Enable saving built parameters file from .nd files")
    mx_group.add_argument("--legacy_mxnet_model",
                          action='store_true',
                          help="Enable MXNet loader to make a model compatible with the latest MXNet version. Use only if your model was trained with MXNet version lower than 1.0.0")
    mx_group.add_argument("--enable_ssd_gluoncv",
                          action='store_true',
                          help="Enable pattern matchers replacers for converting gluoncv ssd topologies.",
                          default=False)

    return parser


def get_kaldi_cli_parser(parser: argparse.ArgumentParser = None):
    """
    Specifies cli arguments for Model Optimizer for MXNet*

    Returns
    -------
        ArgumentParser instance
    """
    if not parser:
        parser = argparse.ArgumentParser()
        get_common_cli_parser(parser=parser)

    kaldi_group = parser.add_argument_group('Kaldi-specific parameters')

    kaldi_group.add_argument("--counts",
                             help="Path to the counts file",
                             default=None,
                             action=CanonicalizePathCheckExistenceAction)

    kaldi_group.add_argument("--remove_output_softmax",
                             help="Removes the SoftMax layer that is the output layer",
                             action='store_true')

    kaldi_group.add_argument("--remove_memory",
                             help="Removes the Memory layer and use additional inputs outputs instead",
                             action='store_true',
                             default=False)
    return parser


def get_onnx_cli_parser(parser: argparse.ArgumentParser = None):
    """
    Specifies cli arguments for Model Optimizer for ONNX

    Returns
    -------
        ArgumentParser instance
    """
    if not parser:
        parser = argparse.ArgumentParser()
        get_common_cli_parser(parser=parser)

    tf_group = parser.add_argument_group('ONNX*-specific parameters')

    return parser


def get_all_cli_parser():
    """
    Specifies cli arguments for Model Optimizer

    Returns
    -------
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--framework',
                        help='Name of the framework used to train the input model.',
                        type=str,
                        choices=['tf', 'caffe', 'mxnet', 'kaldi', 'onnx'])

    get_common_cli_parser(parser=parser)

    get_tf_cli_parser(parser=parser)
    get_caffe_cli_parser(parser=parser)
    get_mxnet_cli_parser(parser=parser)
    get_kaldi_cli_parser(parser=parser)
    get_onnx_cli_parser(parser=parser)

    return parser


def append_exp_keys_to_namespace(argv: argparse.Namespace):
    setattr(argv, 'generate_experimental_IR_V10', False)
    setattr(argv, 'keep_quantize_ops_in_IR', False)
    setattr(argv, 'blobs_as_inputs', False)


def parse_input_value(input_value: str):
    """
    Parses a value of the --input command line parameter and gets a node name, shape and value.
    The node name includes a port if it is specified.
    Shape and value is equal to None if they are not specified.
    Parameters
    ----------
    input_value
        string with a specified node name, shape and value.
        E.g. 'node_name:0[4]->[1.0 2.0 3.0 4.0]'

    Returns
    -------
        Node name, shape and value
        E.g. 'node_name:0', '4', [1.0 2.0 3.0 4.0]
    """
    input_value = input_value.split('->')
    node_name_with_shape = input_value[0]

    # parse a node name
    node_name = node_name_with_shape.split('[')[0]

    # parse shape
    shape = re.findall(r'[(\[]([0-9  -]+)[)\]]', node_name_with_shape)
    if len(shape) == 0:
        shape = None
    elif len(shape) == 1:
        shape = np.fromstring(shape[0], dtype=np.int64, sep=' ')
    else:
        raise Error("Wrong syntax to specify shape. Use --input "
                    "\"node_name[shape]->value\"")

    # parse value, compute shape and check it
    value = None
    if len(input_value) == 2:
        value = input_value[1]
        # check format of value
        if value[0] == '[' and value[-1] != ']' or value[0] != '[' and value[-1] == ']':
            raise Error("Wrong syntax to specify value. Use --input "
                        "\"node_name[shape]->value\"")
        # TODO: support multidimensional value: check format, compute value shape
        # and check that value shape is equal to shape
        if '[' in value.strip(' '):
            value = value.replace('[', '').replace(']', '').split(' ')
        value_size = len(value)
        if shape is not None and np.prod(shape) != value_size:
            raise Error("Wrong syntax to specify value.")
    elif len(input_value) > 2:
        raise Error("Wrong syntax to specify value. Use --input "
                    "\"node_name[shape]->value\"")

    return node_name, shape, value


def get_freeze_placeholder_values(argv_input: str, argv_freeze_placeholder_with_value: str):
    """
    Parses values for placeholder freezing and input node names

    Parameters
    ----------
    argv_input
        string with a list of input layers: either an empty string, or strings separated with comma.
        'node_name1[shape1]->value1,node_name2[shape2]->value2,...'
    argv_freeze_placeholder_with_value
        string with a list of input shapes: either an empty string, or tuples separated with comma.
        'placeholder_name1->value1, placeholder_name2->value2,...'

    Returns
    -------
        parsed placeholders with values for freezing
        input nodes cleaned from shape info

    """
    placeholder_values = {}
    input_node_names = None

    if argv_freeze_placeholder_with_value is not None:
        for plh_with_value in argv_freeze_placeholder_with_value.split(','):
            plh_with_value = plh_with_value.split('->')
            if len(plh_with_value) != 2:
                raise Error("Wrong replacement syntax. Use --freeze_placeholder_with_value "
                            "\"node1_name->value1,node2_name->value2\"")
            node_name = plh_with_value[0]
            value = plh_with_value[1]
            if node_name in placeholder_values and placeholder_values[node_name] != value:
                raise Error("Overriding replacement value of the placeholder with name '{}': old value = {}, new value = {}"
                            ".".format(node_name, placeholder_values[node_name], value))
            if '[' in value.strip(' '):
                value = value.replace('[', '').replace(']', '').split(' ')
            placeholder_values[node_name] = value

    if argv_input is not None:
        input_node_names = ''
        # walkthrough all input values and save values for freezing
        for input_value in argv_input.split(','):
            node_name, _, value = parse_input_value(input_value)
            input_node_names = input_node_names + ',' + node_name  if input_node_names != '' else node_name
            if value is None: # no value is specified for freezing
                continue
            if node_name in placeholder_values and placeholder_values[node_name] != value:
                raise Error("Overriding replacement value of the placeholder with name '{}': old value = {}, new value = {}"
                            ".".format(node_name, placeholder_values[node_name], value))
            placeholder_values[node_name] = value

    return placeholder_values, input_node_names


def get_placeholder_shapes(argv_input: str, argv_input_shape: str, argv_batch=None):
    """
    Parses input layers names and input shapes from the cli and returns the parsed object.
    All shapes are specified only through one command line option either --input or --input_shape.

    Parameters
    ----------
    argv_input
        string with a list of input layers: either an empty string, or strings separated with comma.
        E.g. 'inp1,inp2', 'node_name1[shape1]->value1,node_name2[shape2]->value2'
    argv_input_shape
        string with a list of input shapes: either an empty string, or tuples separated with comma.
        E.g. '(1,2),(3,4)'.
        Only positive integers are accepted except -1, which can be on any position in a shape.
    argv_batch
        integer that overrides batch size in input shape

    Returns
    -------
        parsed shapes in form of {'name of input':ndarray} if names of inputs are provided with shapes
        parsed shapes in form of {'name of input':None} if names of inputs are provided without shapes
        ndarray if only one shape is provided and no input name
        None if neither shape nor input were provided
    """
    if argv_input_shape and argv_batch:
        raise Error("Both --input_shape and --batch were provided. Please provide only one of them. " +
                    refer_to_faq_msg(56))

    # attempt to extract shapes from --input parameters
    placeholder_shapes = dict()
    are_shapes_specified_through_input = False
    if argv_input:
        for input_value in argv_input.split(','):
            node_name, shape, _ = parse_input_value(input_value)
            placeholder_shapes[node_name] = shape
            if shape is not None:
                are_shapes_specified_through_input = True

    if argv_input_shape and are_shapes_specified_through_input:
        raise Error("Shapes are specified using both --input and --input_shape command-line parameters, but only one parameter is allowed.")

    if argv_batch and are_shapes_specified_through_input:
        raise Error("Shapes are specified using both --input and --batch command-line parameters, but only one parameter is allowed.")

    if are_shapes_specified_through_input:
        return placeholder_shapes

    shapes = list()
    inputs = list()
    placeholder_shapes = None

    first_digit_reg = r'([0-9 ]+|-1)'
    next_digits_reg = r'(,{})*'.format(first_digit_reg)
    tuple_reg = r'((\({}{}\))|(\[{}{}\]))'.format(first_digit_reg, next_digits_reg,
                                                  first_digit_reg, next_digits_reg)
    if argv_input_shape:
        full_reg = r'^{}(\s*,\s*{})*$|^$'.format(tuple_reg, tuple_reg)
        if not re.match(full_reg, argv_input_shape):
            raise Error('Input shape "{}" cannot be parsed. ' +
                        refer_to_faq_msg(57), argv_input_shape)

        shapes = re.findall(r'[(\[]([0-9, -]+)[)\]]', argv_input_shape)

    if argv_input:
        inputs = argv_input.split(',')

    # check number of shapes with no input provided
    if argv_input_shape and not argv_input:
        if len(shapes) > 1:
            raise Error('Please provide input layer names for input layer shapes. ' +
                        refer_to_faq_msg(58))
        else:
            placeholder_shapes = np.fromstring(shapes[0], dtype=np.int64, sep=',')

    # check if number of shapes does not match number of passed inputs
    elif argv_input and (len(shapes) == len(inputs) or len(shapes) == 0):
        # clean inputs from values for freezing
        inputs = list(map(lambda x: x.split('->')[0], inputs))
        placeholder_shapes = dict(zip_longest(inputs,
                                              map(lambda x: np.fromstring(x, dtype=np.int64,
                                                                          sep=',') if x else None, shapes)))
    elif argv_input:
        raise Error('Please provide each input layers with an input layer shape. ' +
                    refer_to_faq_msg(58))

    return placeholder_shapes


def parse_tuple_pairs(argv_values: str):
    """
    Gets mean/scale values from the given string parameter
    Parameters
    ----------
    argv_values
        string with a specified input name and  list of mean values: either an empty string, or a tuple
        in a form [] or ().
        E.g. 'data(1,2,3)' means 1 for the RED channel, 2 for the GREEN channel, 3 for the BLUE channel for the data
        input layer, or tuple of values in a form [] or () if input is specified separately, e.g. (1,2,3),[4,5,6].

    Returns
    -------
        dictionary with input name and tuple of values or list of values if mean/scale value is specified with input,
        e.g.:
        "data(10,20,30),info(11,22,33)" -> { 'data': [10,20,30], 'info': [11,22,33] }
        "(10,20,30),(11,22,33)" -> [np.array(10,20,30), np.array(11,22,33)]
    """
    res = {}
    if not argv_values:
        return res

    data_str = argv_values
    while True:
        tuples_matches = re.findall(r'[(\[]([0-9., -]+)[)\]]', data_str, re.IGNORECASE)
        if not tuples_matches :
            raise Error(
                "Mean/scale values should be in format: data(1,2,3),info(2,3,4)" +
                " or just plain set of them without naming any inputs: (1,2,3),(2,3,4). " +
                refer_to_faq_msg(101), argv_values)
        tuple_value = tuples_matches[0]
        matches = data_str.split(tuple_value)

        input_name = matches[0][:-1]
        if not input_name:
            res = []
            # check that other values are specified w/o names
            words_reg = r'([a-zA-Z]+)'
            for i in range(0, len(matches)):
                if re.search(words_reg, matches[i]) is not None:
                    # error - tuple with name is also specified
                    raise Error(
                        "Mean/scale values should either contain names of input layers: data(1,2,3),info(2,3,4)" +
                        " or just plain set of them without naming any inputs: (1,2,3),(2,3,4)." +
                        refer_to_faq_msg(101), argv_values)
            for match in tuples_matches:
                res.append(np.fromstring(match, dtype=float, sep=','))
            break

        res[input_name] = np.fromstring(tuple_value, dtype=float, sep=',')

        parenthesis = matches[0][-1]
        sibling = ')' if parenthesis == '(' else ']'
        pair = '{}{}{}{}'.format(input_name, parenthesis, tuple_value, sibling)
        idx_substr = data_str.index(pair)
        data_str = data_str[idx_substr + len(pair) + 1:]

        if not data_str:
            break

    return res


def get_tuple_values(argv_values: str or tuple, num_exp_values: int = 3, t=float or int):
    """
    Gets mean values from the given string parameter
    Args:
        argv_values: string with list of mean values: either an empty string, or a tuple in a form [] or ().
        E.g. '(1,2,3)' means 1 for the RED channel, 2 for the GREEN channel, 4 for the BLUE channel.
        t: either float or int
        num_exp_values: number of values in tuple

    Returns:
        tuple of values
    """

    digit_reg = r'(-?[0-9. ]+)' if t == float else r'(-?[0-9 ]+)'

    assert num_exp_values > 1, 'Can not parse tuple of size 1'
    content = r'{0}\s*,{1}\s*{0}'.format(digit_reg, (digit_reg + ',') * (num_exp_values - 2))
    tuple_reg = r'((\({0}\))|(\[{0}\]))'.format(content)

    if isinstance(argv_values, tuple) and not len(argv_values):
        return argv_values

    if not len(argv_values) or not re.match(tuple_reg, argv_values):
        raise Error('Values "{}" cannot be parsed. ' +
                    refer_to_faq_msg(59), argv_values)

    mean_values_matches = re.findall(r'[(\[]([0-9., -]+)[)\]]', argv_values)

    for mean in mean_values_matches:
        if len(mean.split(',')) != num_exp_values:
            raise Error('{} channels are expected for given values. ' +
                        refer_to_faq_msg(60), num_exp_values)

    return mean_values_matches


def get_mean_scale_dictionary(mean_values, scale_values, argv_input: str):
    """
    This function takes mean_values and scale_values, checks and processes them into convenient structure

    Parameters
    ----------
    mean_values dictionary, contains input name and mean values passed py user (e.g. {data: np.array[102.4, 122.1, 113.9]}),
    or list containing values (e.g. np.array[102.4, 122.1, 113.9])
    scale_values dictionary, contains input name and scale values passed py user (e.g. {data: np.array[102.4, 122.1, 113.9]})
    or list containing values (e.g. np.array[102.4, 122.1, 113.9])

    Returns
    -------
    The function returns a dictionary e.g.
    mean = { 'data: np.array, 'info': np.array }, scale = { 'data: np.array, 'info': np.array }, input = "data, info" ->
     { 'data': { 'mean': np.array, 'scale': np.array }, 'info': { 'mean': np.array, 'scale': np.array } }

    """
    res = {}
    # collect input names
    if argv_input:
        inputs = argv_input.split(',')
    else:
        inputs = []
        if type(mean_values) is dict:
            inputs = list(mean_values.keys())
        if type(scale_values) is dict:
            for name in scale_values.keys():
                if name not in inputs:
                    inputs.append(name)

    # create unified object containing both mean and scale for input
    if type(mean_values) is dict and type(scale_values) is dict:
        if not mean_values and not scale_values:
            return res
        for inp in inputs:
            inp, port = split_node_in_port(inp)
            if inp in mean_values or inp in scale_values:
                res.update(
                    {
                        inp: {
                            'mean':
                                mean_values[inp] if inp in mean_values else None,
                            'scale':
                                scale_values[inp] if inp in scale_values else None
                        }
                    }
                )
        return res

    # user specified input and mean/scale separately - we should return dictionary
    if inputs:
        if mean_values and scale_values:
            if len(inputs) != len(mean_values):
                raise Error('Numbers of inputs and mean values do not match. ' +
                            refer_to_faq_msg(61))
            if len(inputs) != len(scale_values):
                raise Error('Numbers of inputs and scale values do not match. ' +
                            refer_to_faq_msg(62))

            data = list(zip(mean_values, scale_values))

            for i in range(len(data)):
                res.update(
                    {
                        inputs[i]: {
                            'mean':
                                data[i][0],
                            'scale':
                                data[i][1],

                        }
                    }
                )
            return res
        # only mean value specified
        if mean_values:
            data = list(mean_values)
            for i in range(len(data)):
                res.update(
                    {
                        inputs[i]: {
                            'mean':
                                data[i],
                            'scale':
                                None

                        }
                    }
                )
            return res

        # only scale value specified
        if scale_values:
            data = list(scale_values)
            for i in range(len(data)):
                res.update(
                    {
                        inputs[i]: {
                            'mean':
                                None,
                            'scale':
                                data[i]

                        }
                    }
                )
            return res
    # mean and scale are specified without inputs, return list, order is not guaranteed (?)
    return list(zip_longest(mean_values, scale_values))


def get_model_name(path_input_model: str) -> str:
    """
    Deduces model name by a given path to the input model
    Args:
        path_input_model: path to the input model

    Returns:
        name of the output IR
    """
    parsed_name, extension = os.path.splitext(os.path.basename(path_input_model))
    return 'model' if parsed_name.startswith('.') or len(parsed_name) == 0 else parsed_name


def get_absolute_path(path_to_file: str) -> str:
    """
    Deduces absolute path of the file by a given path to the file
    Args:
        path_to_file: path to the file

    Returns:
        absolute path of the file
    """
    file_path = os.path.expanduser(path_to_file)
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.getcwd(), file_path)
    return file_path


def check_positive(value):
    try:
        int_value = int(value)
        if int_value <= 0:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("expected a positive integer value")

    return int_value


def depersonalize(value: str):
    if not isinstance(value, str):
        return value
    res = []
    for path in value.split(','):
        if os.path.isdir(path):
            res.append('DIR')
        elif os.path.isfile(path):
            res.append(os.path.join('DIR', os.path.split(path)[1]))
        else:
            res.append(path)
    return ','.join(res)


def get_meta_info(argv: argparse.Namespace):
    meta_data = {'unset': []}
    for key, value in argv.__dict__.items():
        if value is not None:
            value = depersonalize(value)
            meta_data[key] = value
        else:
            meta_data['unset'].append(key)
    # The attribute 'k' is treated separately because it points to not existing file by default
    for key in ['k']:
        if key in meta_data:
            meta_data[key] = ','.join([os.path.join('DIR', os.path.split(i)[1]) for i in meta_data[key].split(',')])
    return meta_data

