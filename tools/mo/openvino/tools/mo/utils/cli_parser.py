# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import logging as log
import os
import re
from collections import OrderedDict, namedtuple
from distutils.util import strtobool
from itertools import zip_longest
from pathlib import Path
from operator import xor
from typing import List, Union
import numbers

import numpy as np
from openvino.runtime import Layout, PartialShape, Dimension, Shape, Type

import openvino
from openvino.tools.mo.front.extractor import split_node_in_port
from openvino.tools.mo.middle.passes.convert_data_type import destination_type_to_np_data_type
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from openvino.tools.mo.utils import import_extensions
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg, get_mo_root_dir
from openvino.tools.mo.utils.version import get_version


def extension_path_to_str_or_extensions_class(extension):
    if isinstance(extension, str):
        return extension
    elif isinstance(extension, Path):
        return str(extension)
    else:
        # Return unknown object as is.
        # The type of the object will be checked by frontend.add_extension() method
        return extension


def transformations_config_to_str(value):
    if value is None:
        return value
    return extension_path_to_str_or_extensions_class(value)


def extensions_to_str_or_extensions_class(extensions):
    if extensions is None:
        return [import_extensions.default_path()]
    extensions_list = []
    if isinstance(extensions, str):
        extensions_list = extensions.split(',')
    elif isinstance(extensions, list):
        for ext in extensions:
            ext = extension_path_to_str_or_extensions_class(ext)
            extensions_list.append(ext)
    else:
        extensions_list = [extension_path_to_str_or_extensions_class(extensions)]

    for ext in extensions_list:
        if isinstance(ext, str):
            readable_file_or_dir(ext)
    return extensions_list


def path_to_str(path):
    if path is None:
        return None
    if isinstance(path, str):
        return path
    elif isinstance(path, Path):
        return str(path)
    else:
        raise Exception("Incorrect type of {} expected str or Path, got {}".format(path, type(path)))


def path_to_str_or_object(value):
    if value is None or isinstance(value, str):
        return value
    elif isinstance(value, Path):
        return str(value)
    else:
        return value


def paths_to_str(paths):
    if paths is None:
        return None
    if isinstance(paths, list):
        paths_str = []
        for path in paths:
            paths_str.append(path_to_str(path))
        return ','.join(paths_str)
    else:
        path_to_str(paths)


def str_list_to_str(values):
    if values is None:
        return None
    if isinstance(values, str):
        return values
    elif isinstance(values, list):
        for value in values:
            if not isinstance(value, str):
                raise Error("Incorrect argument. {} expected to string, got type {}.".format(value, type(value)))
        return ','.join(values)
    else:
        raise Error("Incorrect argument. {} expected to string or list of strings, got type {}.".format(values, type(values)))


def is_shape_type(value):
    if isinstance(value, PartialShape):
        return True
    if isinstance(value, Shape):
        return True
    if isinstance(value, list) or isinstance(value, tuple):
        for dim in value:
            if not (isinstance(dim, Dimension) or isinstance(dim, int)):
                return False
        return True
    return False


def shape_to_str(shape, separator):
    if isinstance(shape, str):
        return shape
    if isinstance(shape, PartialShape):
        return shape.to_string()
    if isinstance(shape, Shape):
        return PartialShape(shape).to_string()
    if isinstance(shape, list) or isinstance(shape, tuple):
        dims = []
        for dim in shape:
            if isinstance(dim, Dimension):
                dims.append(dim.to_string())
            elif isinstance(dim, int):
                dims.append(str(dim))
            else:
                raise Exception("Incorrect type of dimension. Expected Dimension or int, got {}".format(type(dim)))
        return "[" + separator.join(dims) + "]"
    raise Exception("Incorrect shape type. Expected PartialShape, Shape, [Dimension, ...] or [int, ...], "
                    "got {}".format(type(shape)))


def input_shape_to_str(input_shape):
    if input_shape is None or isinstance(input_shape, str):
        return input_shape
    if isinstance(input_shape, list):
        if len(input_shape) > 0 and isinstance(input_shape[0], int) or isinstance(input_shape[0], Dimension):
            # The case when shape is specified as list of int or Dimension
            return shape_to_str(input_shape, ',')
        # The case when list of shapes is specified
        shapes = []
        for shape in input_shape:
            shapes.append(shape_to_str(shape, ','))
        return ','.join(shapes)
    return shape_to_str(input_shape, ',')


def type_to_str(type_obj):
    if isinstance(type_obj, str):
        return type_obj
    if isinstance(type_obj, type):
        return np_data_type_to_destination_type(type_obj)
    if isinstance(type_obj, Type):
        return type_obj.get_type_name()
    raise Exception("Incorrect type. Expected Type or numpy type, got {}".format(type(type_obj)))


def value_to_str(value, separator):
    if isinstance(value, np.ndarray):
        values = []
        for x in np.nditer(value):
            values.append(str(x))
        return "[" + separator.join(values) + "]"
    if isinstance(value, list):
        values = []
        for x in value:
            if not isinstance(x, numbers.Number):
                raise Exception("Incorrect value type. Expected numeric value, got {}".format(type(x)))
            values.append(str(x))
        return "[" + separator.join(values) + "]"
    if isinstance(value, bool):
        return "True" if value else "False"
    raise Exception("Incorrect value type. Expected np.ndarray or list, got {}".format(type(value)))


def single_input_to_str(input):
    if isinstance(input, str):
        return input
    if isinstance(input, openvino.tools.mo.InputCutInfo):
        if not isinstance(input.name, str):
            raise Exception("Input name should be string, got {}".format(input.name))
        input_str = input.name
        assert input_str is not None, "Incorrect InputCutInfo. 'name' should be set."
        if input.shape is not None:
            input_str += shape_to_str(input.shape, " ")
        if input.type is not None:
            input_str += "{" + type_to_str(input.type) + "}"
        if input.value is not None:
            input_str += "->" + value_to_str(input.value, " ")
        return input_str
    if isinstance(input, tuple):
        name = None
        inp_type = None
        shape = None
        for val in input:
            if isinstance(val, str):
                if name is not None:
                    raise Exception("More than one input name provided: {}".format(input))
                name = val
            elif isinstance(val, type) or isinstance(val, Type):
                if inp_type is not None:
                    raise Exception("More than one input type provided: {}".format(input))
                inp_type = type_to_str(val)
            elif is_shape_type(val):
                if shape is not None:
                    raise Exception("More than one input shape provided: {}".format(input))
                shape = shape_to_str(val, " ")
            else:
                raise Exception("Incorrect input parameters provided. Expected input name and "
                                "optionally input type or input shape. Got unknown object: {}".format(val))
        if name is None:
            raise Exception("Input name was not provided for following input {}.".format(input))
        if shape is not None:
            name += shape
        if inp_type is not None:
            name += "{" + inp_type + "}"
        return name

    raise Exception("Unexpected object provided for input. Expected openvino.tools.mo.InputCutInfo "
                    "or tuple or str. Got {}".format(type(input)))


def input_to_str(input):
    if input is None or isinstance(input, str):
        return input
    if isinstance(input, list):
        inputs_str = []
        for inp in input:
            inputs_str.append(single_input_to_str(inp))
        return ','.join(inputs_str)
    return single_input_to_str(input)


def mean_scale_value_to_str(value):
    # default empty value
    if isinstance(value, tuple) and len(value) == 0:
        return value

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        values_str = []
        for op_name, val in value.items():
            if not isinstance(op_name, str):
                raise Exception("Incorrect operation name type. Expected string, got {}".format(type(op_name)))
            values_str.append(op_name + value_to_str(val, ","))
        return ",".join(values_str)
    if isinstance(value, list) or isinstance(value, tuple):
        list_of_lists = False
        for val in value:
            if isinstance(val, list) or isinstance(val, tuple):
                list_of_lists = True
                break
        if list_of_lists:
            values_str = []
            for val in value:
                values_str.append(value_to_str(val, ","))
            return ",".join(values_str)
        else:
            return value_to_str(value, ",")
    return value_to_str(value, ",")


def layout_to_str(layout):
    if isinstance(layout, str):
        return layout
    if isinstance(layout, Layout):
        return layout.to_string()
    raise Exception("Incorrect layout type. Expected Layout or string or dictionary, "
                    "where key is operation name and value is layout or list of layouts, got {}".format(type(layout)))


def source_target_layout_to_str(value):
    # default empty value
    if isinstance(value, tuple) and len(value) == 0:
        return value

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        values_str = []
        for op_name, layout in value.items():
            if not isinstance(op_name, str):
                raise Exception("Incorrect operation name type. Expected string, got {}".format(type(op_name)))
            values_str.append(op_name + "(" + layout_to_str(layout) + ")")
        return ",".join(values_str)

    return layout_to_str(value)


def layoutmap_to_str(value):
    if isinstance(value, str):
        return value
    if isinstance(value, openvino.tools.mo.LayoutMap):
        assert value.source_layout is not None, "Incorrect layout map. 'source_layout' should be set."
        source_layout = layout_to_str(value.source_layout)
        if value.target_layout is not None:
            target_layout = layout_to_str(value.target_layout)
            source_layout += "->" + target_layout
        return source_layout
    return layout_to_str(value)


def layout_param_to_str(value):
    # default empty value
    if isinstance(value, tuple) and len(value) == 0:
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        values_str = []
        for op_name, layout in value.items():
            if not isinstance(op_name, str):
                raise Exception("Incorrect operation name type. Expected string, got {}".format(type(op_name)))
            values_str.append(op_name + "(" + layoutmap_to_str(layout) + ")")
        return ",".join(values_str)
    if isinstance(value, openvino.tools.mo.LayoutMap):
        return layoutmap_to_str(value)
    if isinstance(value, list) or isinstance(value, tuple):
        values_str = []
        for layout in value:
            values_str.append(layoutmap_to_str(layout))
        return ",".join(values_str)

    return layoutmap_to_str(value)


def batch_to_int(value):
    if value is None or isinstance(value, int):
        return value
    if isinstance(value, Dimension):
        if not value.is_static:
            # TODO: Ticket 88676
            raise Exception("Dynamic batch for --batch parameter is not supported.")
        else:
            return value.get_length()
    raise Exception("Incorrect batch value. Expected int, got {}.".format(type(value)))


def transform_param_value_to_str(value):
    # This function supports parsing of parameters of MakeStateful, LowLatency2, Pruning.
    # If available transforms list is extended this method should be extended for new transforms.
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, dict):
        # param_res_names dictionary for MakeStateful transform
        values_str = []
        for input_name, output_name in value.items():
            assert isinstance(input_name, str), "Incorrect input name. " \
                                                "Expected string, got {}".format(type(input_name))
            assert isinstance(output_name, str), "Incorrect output name. " \
                                                 "Expected string, got {}".format(type(output_name))
            values_str.append("\'{}\':\'{}\'".format(input_name, output_name))
        return "{" + ','.join(values_str) + "}"
    raise Exception("Unknown parameter type.")


def transform_to_str(value):
    from openvino.tools.mo.back.offline_transformations import get_available_transformations

    if isinstance(value, str):
        return value

    if isinstance(value, tuple):
        assert 1 <= len(value) <= 2, "Incorrect definition of transformation in transform argument: " \
                                     "expected two elements in tuple, provided {}. " \
                                     "Supported transforms are: {}".format(
            len(value),
            list(get_available_transformations().keys()))
        transform_name = value[0]
        assert isinstance(transform_name, str), "Incorrect transform name type. " \
                                                "Expected string, got {}".format(type(transform_name))
        if len(value) == 2:
            params = value[1]
            assert isinstance(params, dict), "Incorrect transform params type. " \
                                             "Expected dictionary, got {}".format(type(params))
            params_str_list = []
            for param_name, val in params.items():
                assert isinstance(param_name, str), "Incorrect transform parameter name type. " \
                                                    "Expected string, got {}".format(type(param_name))
                val_str = transform_param_value_to_str(val)
                params_str_list.append(param_name + "=" + val_str)
            transform_name += '[' + ','.join(params_str_list) + ']'
        return transform_name
    raise Exception("Incorrect transform type. Expected tuple with transform name and "
                    "dictionary with transform parameters. Got object of type {}".format(type(value)))


def transform_param_to_str(value):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        transforms_str = []
        for transform in value:
            transforms_str.append(transform_to_str(transform))
        return ','.join(transforms_str)
    return transform_to_str(value)


ParamDescription = namedtuple("ParamData",
                              ["description", "possible_types_command_line", "possible_types_python_api", "to_string"])
mo_convert_params = {
    'optional':
    {
    'help': ParamDescription(
        'Print available parameters.', '', '', None),
    'framework': ParamDescription(
        'Name of the framework used to train the input model.', '', '', None),
    },
    'fw_agnostic':
    {
    'input_model': ParamDescription(
        '{} Tensorflow*: a file with a pre-trained model ' +
        ' (binary or text .pb file after freezing).\n' +
        ' Caffe*: a model proto file with model weights', '',
        'Model object in original framework (PyTorch, Tensorflow) or path to model file. \n' +
        'Supported object formats of input model:\n PyTorch - torch.nn.Module, torch.jit.ScriptModule, torch.jit.ScriptFunction' +
        'TF - tf.compat.v1.GraphDef, tf.compat.v1.wrap_function, tf.compat.v1.session\n ' +
        'TF2 / Keras - tf.keras.Model, tf.keras.layers.Layer, tf.function, tf.Module, tf.train.checkpoint, ' +
        'tf.python.training.tracking.base.Trackable for case when it is output from tf.saved_model.load().\n' +
        'File formats examples:\n',
        path_to_str_or_object),
    'model_name': ParamDescription(
        'Model_name parameter passed to the final create_ir transform. ' +
        'This parameter is used to name ' +
        'a network in a generated IR and output .xml/.bin files.', '', '', None),
    'input_shape': ParamDescription(
        'Input shape(s) that should be fed to an input node(s) of the model. {}'
        'Shape is defined as a comma-separated list of integer numbers enclosed in '
        'parentheses or square brackets, for example [1,3,227,227] or (1,227,227,3), where '
        'the order of dimensions depends on the framework input layout of the model. '
        'For example, [N,C,H,W] is used for ONNX* models and [N,H,W,C] for TensorFlow* '
        'models. The shape can contain undefined dimensions (? or -1) and '
        'should fit the dimensions defined in the input '
        'operation of the graph. Boundaries of undefined dimension can be specified with '
        'ellipsis, for example [1,1..10,128,128]. One boundary can be undefined, for '
        'example [1,..100] or [1,3,1..,1..]. If there are multiple inputs in the model, '
        '--input_shape should contain definition of shape for each input separated by a '
        'comma, for example: [1,3,227,227],[2,4] for a model with two inputs with 4D and 2D '
        'shapes. Alternatively, specify shapes with the --input option.', '',
        'Input shapes can be defined by passing a list of objects of type '
        'PartialShape, Shape, [Dimension, ...] or [int, ...] or by a string '
        'of the following format. ', input_shape_to_str),
    'scale': ParamDescription(
        'All input values coming from original network inputs will be ' +
        'divided by this ' +
        'value. When a list of inputs is overridden by the --input ' +
        'parameter, this scale ' +
        'is not applied for any input that does not match with ' +
        'the original input of the model. ' +
        'If both --mean_values and --scale  are specified, ' +
        'the mean is subtracted first and then scale is applied ' +
        'regardless of the order of options in command line.', '', '', None),
    'reverse_input_channels': ParamDescription(
        'Switch the input channels order from RGB to BGR (or vice versa). Applied to '
        'original inputs of the model if and only if a number of channels equals 3. '
        'When --mean_values/--scale_values are also specified, reversing of channels will '
        'be applied to user\'s input data first, so that numbers in --mean_values '
        'and --scale_values go in the order of channels used in the original model. '
        'In other words, if both options are specified, then the data flow in the model '
        'looks as following: '
        'Parameter -> ReverseInputChannels -> Mean apply-> Scale apply -> the original body of the model.',
        '', '', None),
    'log_level': ParamDescription(
        'Logger level', '', '', None),
    'input': ParamDescription(
        '{}Quoted list of comma-separated input nodes names with shapes, data types, '
        'and values for freezing. The order of inputs in converted model is the same as '
        'order of specified operation names. The shape and value are specified as comma-separated '
        'lists. The data type of input node is specified in braces and '
        'can have one of the values: f64 (float64), f32 (float32), f16 (float16), '
        'i64 (int64), i32 (int32), u8 (uint8), boolean (bool). Data type is optional. '
        'If it\'s not specified explicitly then there are two options: '
        'if input node is a parameter, data type is taken from the original node dtype, '
        'if input node is not a parameter, data type is set to f32. '
        'Example, to set `input_1` with shape [1,100], and Parameter node `sequence_len` '
        'with scalar input with value `150`, and boolean input `is_training` with '
        '`False` value use the following format: '
        '"input_1[1,100],sequence_len->150,is_training->False". '
        'Another example, use the following format to set input port 0 of the node '
        '`node_name1` with the shape [3,4] as an input node and freeze output port 1 '
        'of the node `node_name2` with the value [20,15] of the int32 type and shape [2]: '
        '"0:node_name1[3,4],node_name2:1[2]{{i32}}->[20,15]".', '',
        'Input can be set by passing a list of InputCutInfo objects or by a list of tuples. '
        'Each tuple should contain input name and optionally input type or input shape. '
        'Example: input=("op_name", PartialShape([-1, 3, 100, 100]), Type(np.float32)). '
        'Alternatively input can be set by a string or list of strings of the following format. ',
        input_to_str),
    'output': ParamDescription(
        'The name of the output operation of the model or list of names. ' +
        'For TensorFlow*, do not add :0 to this name.'
        'The order of outputs in converted model is the same as order of '
        'specified operation names.', '', '', str_list_to_str),
    'mean_values': ParamDescription(
        'Mean values to be used for the input image per channel. {}' +
        'Values to be provided in the (R,G,B) or [R,G,B] format. ' +
        'Can be defined for desired input of the model, for example: ' +
        '"--mean_values data[255,255,255],info[255,255,255]". ' +
        'The exact meaning and order ' +
        'of channels depend on how the original model was trained.', '',
        'Mean values can be set by passing a dictionary, '
        'where key is input name and value is mean value. '
        'For example mean_values={\'data\':[255,255,255],\'info\':[255,255,255]}. '
        'Or mean values can be set by a string of the following format. ',
        mean_scale_value_to_str),
    'scale_values': ParamDescription(
        'Scale values to be used for the input image per channel. {}' +
        'Values are provided in the (R,G,B) or [R,G,B] format. ' +
        'Can be defined for desired input of the model, for example: ' +
        '"--scale_values data[255,255,255],info[255,255,255]". ' +
        'The exact meaning and order ' +
        'of channels depend on how the original model was trained. ' +
        'If both --mean_values and --scale_values are specified, ' +
        'the mean is subtracted first and then scale is applied ' +
        'regardless of the order of options in command line.', '',
        'Scale values can be set by passing a dictionary, '
        'where key is input name and value is scale value. '
        'For example scale_values={\'data\':[255,255,255],\'info\':[255,255,255]}. '
        'Or scale values can be set by a string of the following format. ',
        mean_scale_value_to_str),
    'source_layout': ParamDescription(
        'Layout of the input or output of the model in the framework. {}Layout can'
        ' be specified in the short form, e.g. nhwc, or in complex form, e.g. "[n,h,w,c]".'
        ' Example for many names: '
        '"in_name1([n,h,w,c]),in_name2(nc),out_name1(n),out_name2(nc)". Layout can be '
        'partially defined, "?" can be used to specify undefined layout for one dimension, '
        '"..." can be used to specify undefined layout for multiple dimensions, for example '
        '"?c??", "nc...", "n...c", etc.', '',
        'Layout can be set by passing a dictionary, where key is input name and value is '
        'LayoutMap object. Or layout can be set by string of the following format. ',
        source_target_layout_to_str),
    'target_layout': ParamDescription(
        'Same as --source_layout, but specifies target layout that will be in the model '
        'after processing by ModelOptimizer.', '', '', source_target_layout_to_str),
    'layout': ParamDescription(
        'Combination of --source_layout and --target_layout. Can\'t be used with either of '
        'them. If model has one input it is sufficient to specify layout of this input, for'
        ' example --layout nhwc. To specify layouts of many tensors, names must be provided,'
        ' for example: --layout "name1(nchw),name2(nc)". It is possible to instruct '
        'ModelOptimizer to change layout, for example: '
        '--layout "name1(nhwc->nchw),name2(cn->nc)". Also "*" in long layout form can be'
        ' used to fuse dimensions, for example "[n,c,...]->[n*c,...]".', '', '', layout_param_to_str),
    'compress_to_fp16': ParamDescription(
        'If the original model has FP32 weights or biases, they are compressed to FP16. '
        'All intermediate data is kept in original precision. Option can be specified alone as "--compress_to_fp16", '
        'or explicit True/False values can be set, for example: "--compress_to_fp16=False", or "--compress_to_fp16=True"',
        '', '', None),
    'transform': ParamDescription(
        'Apply additional transformations. {}' +
        '"--transform transformation_name1[args],transformation_name2..." ' +
        'where [args] is key=value pairs separated by semicolon. ' +
        'Examples:' +
        '          "--transform LowLatency2" or \n' +
        '          "--transform Pruning" or \n' +
        '          "--transform LowLatency2[use_const_initializer=False]" or \n' +
        '          "--transform \"MakeStateful[param_res_names=\n'
        '{{\'input_name_1\':\'output_name_1\',\'input_name_2\':\'output_name_2\'}}]\"" ' +
        'Available transformations: "LowLatency2", "MakeStateful", "Pruning"', 'Usage: ',
        '\'transform\' can be set by a list of tuples, where the first element is '
        'transform name and the second element is transform parameters. '
        'For example: [(\'LowLatency2\', {{\'use_const_initializer\': False}}), ...]',
        transform_param_to_str),
    'extensions': ParamDescription(
        "{} For the legacy MO path (if `--use_legacy_frontend` is used), "
        "a directory or a comma-separated list of directories with extensions are supported. "
        "To disable all extensions including those that are placed at the default location, "
        "pass an empty string.",
        "Paths or a comma-separated list of paths to libraries (.so or .dll) with extensions.",
        "Paths to libraries (.so or .dll) with extensions, comma-separated list of paths, "
        "objects derived from BaseExtension class or lists of objects.",
        extensions_to_str_or_extensions_class),
    'batch': ParamDescription(
        'Input batch size', '', '', batch_to_int),
    'silent': ParamDescription(
        'Prevent any output messages except those that correspond to log level equals '
        'ERROR, that can be set with the following option: --log_level. '
        'By default, log level is already ERROR. ', '', '', None),
    'version': ParamDescription(
        "Version of Model Optimizer", '', '', None
    ),
    'static_shape': ParamDescription(
        'Enables IR generation for fixed input shape (folding `ShapeOf` operations and '
        'shape-calculating sub-graphs to `Constant`). Changing model input shape using '
        'the OpenVINO Runtime API in runtime may fail for such an IR.', '', '', None),
    'progress': ParamDescription(
        'Enable model conversion progress display.', '', '', None),
    'stream_output': ParamDescription(
        'Switch model conversion progress display to a multiline mode.', '', '', None),
    'transformations_config': ParamDescription(
        'Use the configuration file with transformations '
        'description{}. Transformations file can be specified as relative path '
        'from the current directory, as absolute path or as a'
        'relative path from the mo root directory.', '',
        ' or pass object derived from BaseExtension class.',
        transformations_config_to_str),
    'use_new_frontend': ParamDescription(
        'Force the usage of new Frontend of Model Optimizer for model conversion into IR. '
        'The new Frontend is C++ based and is available for ONNX* and PaddlePaddle* models. '
        'Model optimizer uses new Frontend for ONNX* and PaddlePaddle* by default that means '
        '`--use_new_frontend` and `--use_legacy_frontend` options are not specified.', '', '', None),
    'use_legacy_frontend': ParamDescription(
        'Force the usage of legacy Frontend of Model Optimizer for model conversion into IR. '
        'The legacy Frontend is Python based and is available for TensorFlow*, ONNX*, MXNet*, '
        'Caffe*, and Kaldi* models.', '', '', None),
    },
    "caffe":
    {
    'input_proto': ParamDescription(
        'Deploy-ready prototxt file that contains a topology structure ' +
        'and layer attributes', '', '', path_to_str),
    'caffe_parser_path': ParamDescription(
        'Path to Python Caffe* parser generated from caffe.proto', '', '',
        path_to_str),
    'k': ParamDescription(
        'Path to CustomLayersMapping.xml to register custom layers', '', '', path_to_str),
    'disable_omitting_optional': ParamDescription(
        'Disable omitting optional attributes to be used for custom layers. ' +
        'Use this option if you want to transfer all attributes of a custom layer to IR. ' +
        'Default behavior is to transfer the attributes with default values '
        'and the attributes defined by the user to IR.',
        '', '', None),
    'enable_flattening_nested_params': ParamDescription(
        'Enable flattening optional params to be used for custom layers. ' +
        'Use this option if you want to transfer attributes of a custom layer to IR with flattened nested parameters. ' +
        'Default behavior is to transfer the attributes without flattening nested parameters.', '', '', None),
    },
    "tf":
    {
    'input_model_is_text': ParamDescription(
        'TensorFlow*: treat the input model file as a text protobuf format. If not specified, ' +
        'the Model Optimizer treats it as a binary file by default.', '', '', None),
    'input_checkpoint': ParamDescription(
        'TensorFlow*: variables file to load.', '', '', path_to_str),
    'input_meta_graph': ParamDescription(
        'Tensorflow*: a file with a meta-graph of the model before freezing', '', '',
        path_to_str),
    'saved_model_dir': ParamDescription(
        'TensorFlow*: directory with a model in SavedModel format '
        'of TensorFlow 1.x or 2.x version.', '', '', path_to_str),
    'saved_model_tags': ParamDescription(
        "Group of tag(s) of the MetaGraphDef to load, in string format, separated by ','. "
        "For tag-set contains multiple tags, all tags must be passed in.", '', '', str_list_to_str),
    'tensorflow_custom_operations_config_update': ParamDescription(
        'TensorFlow*: update the configuration file with node name patterns with input/output '
        'nodes information.', '', '', path_to_str),
    'tensorflow_object_detection_api_pipeline_config': ParamDescription(
        'TensorFlow*: path to the pipeline configuration file used to generate model created '
        'with help of Object Detection API.', '', '', path_to_str),
    'tensorboard_logdir': ParamDescription(
        'TensorFlow*: dump the input graph to a given directory that should be used with TensorBoard.', '', '',
        path_to_str),
    'tensorflow_custom_layer_libraries': ParamDescription(
        'TensorFlow*: comma separated list of shared libraries with TensorFlow* custom '
        'operations implementation.', '', '', path_to_str),
    },
    "mxnet":
    {
    'input_symbol': ParamDescription(
        'Symbol file (for example, model-symbol.json) that contains a topology structure ' +
        'and layer attributes', '', '', path_to_str),
    'nd_prefix_name': ParamDescription(
        "Prefix name for args.nd and argx.nd files.", '', '', None),
    'pretrained_model_name': ParamDescription(
        "Name of a pretrained MXNet model without extension and epoch number. "
        "This model will be merged with args.nd and argx.nd files",
        '', '', None),
    'save_params_from_nd': ParamDescription(
        "Enable saving built parameters file from .nd files", '', '', None),
    'legacy_mxnet_model': ParamDescription(
        "Enable MXNet loader to make a model compatible with the latest MXNet version. "
        "Use only if your model was trained with MXNet version lower than 1.0.0",
        '', '', None),
    'enable_ssd_gluoncv': ParamDescription(
        "Enable pattern matchers replacers for converting gluoncv ssd topologies.",
        '', '', None),
    },
    "kaldi":
    {
    'counts': ParamDescription(
        "Path to the counts file", '', '', path_to_str),
    'remove_output_softmax': ParamDescription(
        "Removes the SoftMax layer that is the output layer", '', '', None),
    'remove_memory': ParamDescription(
        "Removes the Memory layer and use additional inputs outputs instead", '', '',
        None),
    },
    "pytorch":
    {
    'example_input': ParamDescription('Sample of model input in original framework. '
                                       'For PyTorch it can be torch.Tensor.', '', '', None),
    'onnx_opset_version': ParamDescription('Version of ONNX opset that is used for converting from PyTorch to ONNX.',
                                           '', '', None),
    'input_signature': ParamDescription('PyTorch model forward method input signature, ' 
                                        'will be detected automatically for torch.nn.Module based model instances, '
                                        'for for scripted models may requires to set manually. Example of usage: for forward method defined as'
                                        ' def forward(self, x, y), it will be ["x", "y"]', '', '', None)
    }
}


class DeprecatedStoreTrue(argparse.Action):
    def __init__(self, nargs=0, **kw):
        super().__init__(nargs=nargs, **kw)

    def __call__(self, parser, namespace, values, option_string=None):
        dep_msg = "Use of deprecated cli option {} detected. Option use in the following releases will be fatal. ".format(option_string)
        if 'fusing' in option_string:
            dep_msg += 'Please use --finegrain_fusing cli option instead'
        log.error(dep_msg, extra={'is_warning': True})
        setattr(namespace, self.dest, True)


class DeprecatedOptionCommon(argparse.Action):
    def __call__(self, parser, args, values, option_string):
        dep_msg = "Use of deprecated cli option {} detected. Option use in the following releases will be fatal. ".format(option_string)
        log.error(dep_msg, extra={'is_warning': True})
        setattr(args, self.dest, values)


class IgnoredAction(argparse.Action):
    def __init__(self, nargs=0, **kw):
        super().__init__(nargs=nargs, **kw)

    def __call__(self, parser, namespace, values, option_string=None):
        dep_msg = "Use of removed cli option '{}' detected. The option is ignored. ".format(option_string)
        log.error(dep_msg, extra={'is_warning': True})
        setattr(namespace, self.dest, True)


def canonicalize_and_check_paths(values: Union[str, List[str]], param_name,
                                 try_mo_root=False, check_existance=True) -> List[str]:
    if values is not None:
        list_of_values = list()
        if isinstance(values, str):
            if values != "":
                list_of_values = values.split(',')
        elif isinstance(values, list):
            list_of_values = values
        else:
            raise Error('Unsupported type of command line parameter "{}" value'.format(param_name))

        if not check_existance:
            return [get_absolute_path(path) for path in list_of_values]

        for idx, val in enumerate(list_of_values):
            list_of_values[idx] = val

            error_msg = 'The value for command line parameter "{}" must be existing file/directory, ' \
                        'but "{}" does not exist.'.format(param_name, val)
            if os.path.exists(val):
                continue
            elif not try_mo_root or val == '':
                raise Error(error_msg)
            elif try_mo_root:
                path_from_mo_root = get_mo_root_dir() + '/mo/' + val
                list_of_values[idx] = path_from_mo_root
                if not os.path.exists(path_from_mo_root):
                    raise Error(error_msg)

        return [get_absolute_path(path) for path in list_of_values]


class CanonicalizePathAction(argparse.Action):
    """
    Expand user home directory paths and convert relative-paths to absolute.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        list_of_paths = canonicalize_and_check_paths(values, param_name=option_string,
                                                     try_mo_root=False, check_existance=False)
        setattr(namespace, self.dest, ','.join(list_of_paths))


class CanonicalizeTransformationPathCheckExistenceAction(argparse.Action):
    """
    Convert relative to the current and relative to mo root paths to absolute
    and check specified file or directory existence.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        list_of_paths = canonicalize_and_check_paths(values, param_name=option_string,
                                                     try_mo_root=True, check_existance=True)
        setattr(namespace, self.dest, ','.join(list_of_paths))


class CanonicalizePathCheckExistenceAction(argparse.Action):
    """
    Expand user home directory paths and convert relative-paths to absolute and check specified file or directory
    existence.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        list_of_paths = canonicalize_and_check_paths(values, param_name=option_string,
                                                     try_mo_root=False, check_existance=True)
        setattr(namespace, self.dest, ','.join(list_of_paths))


class CanonicalizeExtensionsPathCheckExistenceAction(argparse.Action):
    """
    Expand user home directory paths and convert relative-paths to absolute and check specified file or directory
    existence.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        list_of_paths = canonicalize_and_check_paths(values, param_name=option_string,
                                                     try_mo_root=False, check_existance=True)
        # Extensions paths are needed to be stored as list
        setattr(namespace, self.dest, list_of_paths)


class CanonicalizePathCheckExistenceIfNeededAction(CanonicalizePathCheckExistenceAction):

    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            if isinstance(values, str):
                if values != "":
                    super().__call__(parser, namespace, values, option_string)
                else:
                    setattr(namespace, self.dest, values)


class DeprecatedCanonicalizePathCheckExistenceAction(CanonicalizePathCheckExistenceAction):
    def __call__(self, parser, namespace, values, option_string=None):
        dep_msg = "Use of deprecated cli option {} detected. Option use in the following releases will be fatal. ".format(
            option_string)
        if 'tensorflow_use_custom_operations_config' in option_string:
            dep_msg += 'Please use --transformations_config cli option instead'
        log.error(dep_msg, extra={'is_warning': True})
        super().__call__(parser, namespace, values, option_string)


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


def readable_file_or_dir(path: str):
    """
    Check that specified path is a readable file or directory.
    :param path: path to check
    :return: path if the file/directory is readable
    """
    if not os.path.isfile(path) and not os.path.isdir(path):
        raise Error('The "{}" is not existing file or directory'.format(path))
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


def readable_dirs_or_files_or_empty(paths: str):
    """
    Checks that comma separated list of paths are readable directories, files or a provided path is empty.
    :param paths: comma separated list of paths.
    :return: comma separated list of paths.
    """
    if paths:
        paths_list = [readable_file_or_dir(path) for path in paths.split(',')]
        return ','.join(paths_list)
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
    mo_convert_params_common = mo_convert_params['fw_agnostic']
    # Common parameters
    common_group.add_argument('--input_model', '-w', '-m',
                              help=mo_convert_params_common['input_model'].description,
                              action=CanonicalizePathCheckExistenceAction,
                              type=readable_file_or_dir)
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
                              help=mo_convert_params_common['input_shape'].description.format(
                                  mo_convert_params_common['input_shape'].possible_types_command_line))
    common_group.add_argument('--scale', '-s',
                              type=float,
                              help='All input values coming from original network inputs will be ' +
                                   'divided by this ' +
                                   'value. When a list of inputs is overridden by the --input ' +
                                   'parameter, this scale ' +
                                   'is not applied for any input that does not match with ' +
                                   'the original input of the model.' +
                                   'If both --mean_values and --scale  are specified, ' +
                                   'the mean is subtracted first and then scale is applied ' +
                                   'regardless of the order of options in command line.')
    common_group.add_argument('--reverse_input_channels',
                              help='Switch the input channels order from RGB to BGR (or vice versa). Applied to '
                                   'original inputs of the model if and only if a number of channels equals 3. '
                                   'When --mean_values/--scale_values are also specified, reversing of channels will '
                                   'be applied to user\'s input data first, so that numbers in --mean_values '
                                   'and --scale_values go in the order of channels used in the original model. '
                                   'In other words, if both options are specified, then the data flow in the model '
                                   'looks as following: Parameter -> ReverseInputChannels -> Mean apply-> Scale apply -> the original body of the model.',
                              action='store_true')
    common_group.add_argument('--log_level',
                              help='Logger level',
                              choices=['CRITICAL', 'ERROR', 'WARN', 'WARNING', 'INFO',
                                       'DEBUG', 'NOTSET'],
                              default='ERROR')
    common_group.add_argument('--input',
                              help=mo_convert_params_common['input'].description.format(
                                  mo_convert_params_common['input'].possible_types_command_line))
    common_group.add_argument('--output',
                              help=mo_convert_params_common['output'].description.format(
                                  mo_convert_params_common['output'].possible_types_command_line))
    common_group.add_argument('--mean_values', '-ms',
                              help=mo_convert_params_common['mean_values'].description.format(
                                  mo_convert_params_common['mean_values'].possible_types_command_line),
                              default=())
    common_group.add_argument('--scale_values',
                              help=mo_convert_params_common['scale_values'].description.format(
                                  mo_convert_params_common['scale_values'].possible_types_command_line),
                              default=())
    common_group.add_argument('--source_layout',
                              help=mo_convert_params_common['source_layout'].description.format(
                                  mo_convert_params_common['source_layout'].possible_types_command_line),
                              default=())
    common_group.add_argument('--target_layout',
                              help=mo_convert_params_common['target_layout'].description.format(
                                  mo_convert_params_common['target_layout'].possible_types_command_line),
                              default=())
    common_group.add_argument('--layout',
                              help=mo_convert_params_common['layout'].description.format(
                                  mo_convert_params_common['layout'].possible_types_command_line),
                              default=())
    # TODO: isn't it a weights precision type
    common_group.add_argument('--data_type',
                              help='[DEPRECATED] Data type for model weights and biases. '
                                   'If original model has FP32 weights or biases and --data_type=FP16 is specified, '
                                   'FP32 model weights and biases are compressed to FP16. '
                                   'All intermediate data is kept in original precision.',
                              choices=["FP16", "FP32", "half", "float"],
                              default='FP16',
                              action=DeprecatedOptionCommon)
    common_group.add_argument('--compress_to_fp16',
                              help=mo_convert_params_common['compress_to_fp16'].description,
                              type=check_bool,
                              nargs="?",
                              const=True,
                              default=True)
    common_group.add_argument('--transform',
                              help=mo_convert_params_common['transform'].description.format(
                                  mo_convert_params_common['transform'].possible_types_command_line),
                              default="")
    # we use CanonicalizeDirCheckExistenceAction instead of readable_dirs to handle empty strings
    common_group.add_argument("--extensions",
                              help=mo_convert_params_common['extensions'].description.format(
                                  mo_convert_params_common['extensions'].possible_types_command_line),
                              default=[import_extensions.default_path()],
                              action=CanonicalizeExtensionsPathCheckExistenceAction,
                              type=readable_dirs_or_files_or_empty)
    common_group.add_argument("--batch", "-b",
                              type=check_positive,
                              default=None,
                              help=mo_convert_params_common['batch'].description)
    common_group.add_argument("--version",
                              action='version',
                              version='Version of Model Optimizer is: {}'.format(get_version()),
                              help=mo_convert_params_common['version'].description)

    common_group.add_argument('--silent',
                              help=mo_convert_params_common['silent'].description,
                              type=check_bool,
                              default=True)
    common_group.add_argument('--freeze_placeholder_with_value',
                              help='Replaces input layer with constant node with '
                                   'provided value, for example: "node_name->True". '
                                   'It will be DEPRECATED in future releases. '
                                   'Use --input option to specify a value for freezing.',
                              default=None)
    common_group.add_argument('--static_shape',
                              help=mo_convert_params_common['static_shape'].description,
                              action='store_true', default=False)
    common_group.add_argument('--progress',
                              help=mo_convert_params_common['progress'].description,
                              action='store_true', default=False)
    common_group.add_argument('--stream_output',
                              help=mo_convert_params_common['stream_output'].description,
                              action='store_true', default=False)
    common_group.add_argument('--transformations_config',
                              help=mo_convert_params_common['transformations_config'].description.format(
                                  mo_convert_params_common['transformations_config'].possible_types_command_line),
                              action=CanonicalizeTransformationPathCheckExistenceAction)
    common_group.add_argument("--use_new_frontend",
                              help=mo_convert_params_common['use_new_frontend'].description,
                              action='store_true', default=False)
    common_group.add_argument("--use_legacy_frontend",
                              help=mo_convert_params_common['use_legacy_frontend'].description,
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
    d['source_layout'] = ['- Source layout', lambda x: x if x else 'Not specified']
    d['target_layout'] = ['- Target layout', lambda x: x if x else 'Not specified']
    d['layout'] = ['- Layout', lambda x: x if x else 'Not specified']
    d['mean_values'] = ['- Mean values', lambda x: x if x else 'Not specified']
    d['scale_values'] = ['- Scale values', lambda x: x if x else 'Not specified']
    d['scale'] = ['- Scale factor', lambda x: x if x else 'Not specified']
    d['data_type'] = ['- Precision of IR', lambda x: 'FP32' if x == 'float' else 'FP16' if x == 'half' else x]
    d['transform'] = ['- User transformations', lambda x: x if x else 'Not specified']
    d['reverse_input_channels'] = '- Reverse input channels'
    d['static_shape'] = '- Enable IR generation for fixed input shape'
    d['transformations_config'] = '- Use the transformations config file'
    return d


def get_advanced_cli_options():
    d = OrderedDict()
    d['use_legacy_frontend'] = '- Force the usage of legacy Frontend of Model Optimizer for model conversion into IR'
    d['use_new_frontend'] = '- Force the usage of new Frontend of Model Optimizer for model conversion into IR'
    return d


def get_caffe_cli_options():
    d = {
        'input_proto': ['- Path to the Input prototxt', lambda x: x],
        'caffe_parser_path': ['- Path to Python Caffe* parser generated from caffe.proto', lambda x: x],
        'k': '- Path to CustomLayersMapping.xml',
    }

    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def get_tf_cli_options():
    d = {
        'input_model_is_text': '- Input model in text protobuf format',
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
        'legacy_mxnet_model': '- Enable MXNet loader for models trained with MXNet version lower than 1.0.0',
    }

    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def get_kaldi_cli_options():
    d = {
        'counts': '- A file name with full path to the counts file or empty string if you want to use counts from model',
        'remove_output_softmax': '- Removes the SoftMax layer that is the output layer',
        'remove_memory': '- Removes the Memory layer and use additional inputs and outputs instead'
    }

    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def get_onnx_cli_options():
    d = {
    }

    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def get_params_with_paths_list():
    return ['input_model', 'output_dir', 'caffe_parser_path', 'extensions', 'k', 'output_dir',
            'input_checkpoint', 'input_meta_graph', 'input_proto', 'input_symbol',
            'pretrained_model_name', 'saved_model_dir', 'tensorboard_logdir',
            'tensorflow_custom_layer_libraries', 'tensorflow_custom_operations_config_update',
            'tensorflow_object_detection_api_pipeline_config', 'tensorflow_use_custom_operations_config',
            'transformations_config']


def get_caffe_cli_parser(parser: argparse.ArgumentParser = None):
    """
    Specifies cli arguments for Model Optimizer for Caffe*

    Returns
    -------
        ArgumentParser instance
    """
    if not parser:
        parser = argparse.ArgumentParser(usage='%(prog)s [options]')
        get_common_cli_parser(parser=parser)

    caffe_group = parser.add_argument_group('Caffe*-specific parameters')
    mo_convert_params_caffe = mo_convert_params['caffe']

    caffe_group.add_argument('--input_proto', '-d',
                             help=mo_convert_params_caffe['input_proto'].description,
                             type=str,
                             action=CanonicalizePathCheckExistenceAction)
    caffe_group.add_argument('--caffe_parser_path',
                             help=mo_convert_params_caffe['caffe_parser_path'].description,
                             type=str,
                             default=os.path.join(os.path.dirname(__file__), os.pardir, 'front', 'caffe', 'proto'),
                             action=CanonicalizePathCheckExistenceAction)
    caffe_group.add_argument('-k',
                             help=mo_convert_params_caffe['k'].description,
                             type=str,
                             default=os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'extensions',
                                                  'front', 'caffe',
                                                  'CustomLayersMapping.xml'),
                             action=CanonicalizePathCheckExistenceAction)
    caffe_group.add_argument('--disable_omitting_optional',
                             help=mo_convert_params_caffe['disable_omitting_optional'].description,
                             action='store_true',
                             default=False)
    caffe_group.add_argument('--enable_flattening_nested_params',
                             help=mo_convert_params_caffe['enable_flattening_nested_params'].description,
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
        parser = argparse.ArgumentParser(usage='%(prog)s [options]')
        get_common_cli_parser(parser=parser)
    mo_convert_params_tf = mo_convert_params['tf']

    tf_group = parser.add_argument_group('TensorFlow*-specific parameters')
    tf_group.add_argument('--input_model_is_text',
                          help=mo_convert_params_tf['input_model_is_text'].description,
                          action='store_true')
    tf_group.add_argument('--input_checkpoint', type=str, default=None,
                          help=mo_convert_params_tf['input_checkpoint'].description,
                          action=CanonicalizePathCheckExistenceAction)
    tf_group.add_argument('--input_meta_graph',
                          help=mo_convert_params_tf['input_meta_graph'].description,
                          action=CanonicalizePathCheckExistenceAction,
                          type=readable_file)
    tf_group.add_argument('--saved_model_dir', default=None,
                          help=mo_convert_params_tf['saved_model_dir'].description,
                          action=CanonicalizePathCheckExistenceAction,
                          type=readable_dirs)
    tf_group.add_argument('--saved_model_tags', type=str, default=None,
                          help=mo_convert_params_tf['saved_model_tags'].description)
    tf_group.add_argument('--tensorflow_custom_operations_config_update',
                          help=mo_convert_params_tf['tensorflow_custom_operations_config_update'].description,
                          action=CanonicalizePathCheckExistenceAction)
    tf_group.add_argument('--tensorflow_use_custom_operations_config',
                          help='Use the configuration file with custom operation description.',
                          action=DeprecatedCanonicalizePathCheckExistenceAction)
    tf_group.add_argument('--tensorflow_object_detection_api_pipeline_config',
                          help=mo_convert_params_tf['tensorflow_object_detection_api_pipeline_config'].description,
                          action=CanonicalizePathCheckExistenceAction)
    tf_group.add_argument('--tensorboard_logdir',
                          help=mo_convert_params_tf['tensorboard_logdir'].description,
                          default=None,
                          action=CanonicalizePathCheckExistenceAction)
    tf_group.add_argument('--tensorflow_custom_layer_libraries',
                          help=mo_convert_params_tf['tensorflow_custom_layer_libraries'].description,
                          default=None,
                          action=CanonicalizePathCheckExistenceAction)
    tf_group.add_argument('--disable_nhwc_to_nchw',
                          help='[DEPRECATED] Disables the default translation from NHWC to NCHW. Since 2022.1 this option '
                               'is deprecated and used only to maintain backward compatibility with previous releases.',
                          action=DeprecatedStoreTrue, default=False)
    return parser


def get_mxnet_cli_parser(parser: argparse.ArgumentParser = None):
    """
    Specifies cli arguments for Model Optimizer for MXNet*

    Returns
    -------
        ArgumentParser instance
    """
    if not parser:
        parser = argparse.ArgumentParser(usage='%(prog)s [options]')
        get_common_cli_parser(parser=parser)

    mx_group = parser.add_argument_group('Mxnet-specific parameters')
    mo_convert_params_mxnet = mo_convert_params['mxnet']

    mx_group.add_argument('--input_symbol',
                          help=mo_convert_params_mxnet['input_symbol'].description,
                          type=str,
                          action=CanonicalizePathCheckExistenceAction)
    mx_group.add_argument("--nd_prefix_name",
                          help=mo_convert_params_mxnet['nd_prefix_name'].description,
                          default=None)
    mx_group.add_argument("--pretrained_model_name",
                          help=mo_convert_params_mxnet['pretrained_model_name'].description,
                          default=None)
    mx_group.add_argument("--save_params_from_nd",
                          action='store_true',
                          help=mo_convert_params_mxnet['save_params_from_nd'].description)
    mx_group.add_argument("--legacy_mxnet_model",
                          action='store_true',
                          help=mo_convert_params_mxnet['legacy_mxnet_model'].description)
    mx_group.add_argument("--enable_ssd_gluoncv",
                          action='store_true',
                          help=mo_convert_params_mxnet['enable_ssd_gluoncv'].description,
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
        parser = argparse.ArgumentParser(usage='%(prog)s [options]')
        get_common_cli_parser(parser=parser)

    kaldi_group = parser.add_argument_group('Kaldi-specific parameters')
    mo_convert_params_kaldi = mo_convert_params['kaldi']

    kaldi_group.add_argument("--counts",
                             help=mo_convert_params_kaldi['counts'].description,
                             default=None,
                             action=CanonicalizePathCheckExistenceIfNeededAction)

    kaldi_group.add_argument("--remove_output_softmax",
                             help=mo_convert_params_kaldi['remove_output_softmax'].description,
                             action='store_true',
                             default=False)

    kaldi_group.add_argument("--remove_memory",
                             help=mo_convert_params_kaldi['remove_memory'].description,
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
        parser = argparse.ArgumentParser(usage='%(prog)s [options]')
        get_common_cli_parser(parser=parser)

    return parser


def get_all_cli_parser(frontEndManager=None):
    """
    Specifies cli arguments for Model Optimizer

    Returns
    -------
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')

    frameworks = list(set(['tf', 'caffe', 'mxnet', 'kaldi', 'onnx'] +
                          (get_available_front_ends(frontEndManager) if frontEndManager else [])))

    parser.add_argument('--framework',
                        help='Name of the framework used to train the input model.',
                        type=str,
                        choices=frameworks)

    get_common_cli_parser(parser=parser)

    get_tf_cli_parser(parser=parser)
    get_caffe_cli_parser(parser=parser)
    get_mxnet_cli_parser(parser=parser)
    get_kaldi_cli_parser(parser=parser)
    get_onnx_cli_parser(parser=parser)

    return parser


def remove_data_type_from_input_value(input_value: str):
    """
    Removes the type specification from the input string. The type specification is a string enclosed with curly braces.
    :param input_value: string passed as input to the --input command line parameter
    :return: string without type specification
    """
    return re.sub(r'\{.*\}', '', input_value)


def get_data_type_from_input_value(input_value: str):
    """
    Returns the numpy data type corresponding to the data type specified in the input value string
    :param input_value: string passed as input to the --input command line parameter
    :return: the corresponding numpy data type and None if the data type is not specified in the input value
    """
    data_type_match = re.match(r'.*\{(.*)\}.*', input_value)
    return destination_type_to_np_data_type(data_type_match.group(1)) if data_type_match is not None else None


def remove_shape_from_input_value(input_value: str):
    """
    Removes the shape specification from the input string. The shape specification is a string enclosed with square
    brackets.
    :param input_value: string passed as input to the --input command line parameter
    :return: string without shape specification
    """
    assert '->' not in input_value, 'The function should not be called for input_value with constant value specified'
    return re.sub(r'[(\[]([0-9\.?,  -]*)[)\]]', '', input_value)


def get_shape_from_input_value(input_value: str):
    """
    Returns PartialShape corresponding to the shape specified in the input value string
    :param input_value: string passed as input to the --input command line parameter
    :return: the corresponding shape and None if the shape is not specified in the input value
    """
    # remove the tensor value from the input_value first
    input_value = input_value.split('->')[0]

    # parse shape
    shape = re.findall(r'[(\[]([0-9\.\?,  -]*)[)\]]', input_value)
    if len(shape) == 0:
        shape = None
    elif len(shape) == 1 and shape[0] in ['', ' ']:
        # this shape corresponds to scalar
        shape = PartialShape([])
    elif len(shape) == 1:
        dims = re.split(r', *| +', shape[0])
        dims = list(filter(None, dims))
        shape = PartialShape([Dimension(dim) for dim in dims])
    else:
        raise Error("Wrong syntax to specify shape. Use --input "
                    "\"node_name[shape]->value\"")
    return shape


def get_node_name_with_port_from_input_value(input_value: str):
    """
    Returns the node name (optionally with input/output port) from the input value
    :param input_value: string passed as input to the --input command line parameter
    :return: the corresponding node name with input/output port
    """
    return remove_shape_from_input_value(remove_data_type_from_input_value(input_value.split('->')[0]))


def get_value_from_input_value(input_value: str):
    """
    Returns the value from the input value string
    :param input_value: string passed as input to the --input command line parameter
    :return: the corresponding value or None if it is not specified
    """
    parts = input_value.split('->')
    value = None
    if len(parts) == 2:
        value = parts[1]
        if value[0] == '[' and value[-1] != ']' or value[0] != '[' and value[-1] == ']':
            raise Error("Wrong syntax to specify value. Use --input \"node_name[shape]->value\"")
        if '[' in value.strip(' '):
            value = value.replace('[', '').replace(']', '')
            if ',' in value:
                value = value.replace(' ', '')
                value = value.split(',')
            else:
                value = value.split(' ')
        if not isinstance(value, list):
            value = ast.literal_eval(value)
    elif len(parts) > 2:
        raise Error("Wrong syntax to specify value. Use --input \"node_name[shape]->value\"")
    return value


def partial_shape_prod(shape: [PartialShape, tuple]):
    assert not (isinstance(shape, PartialShape) and shape.is_dynamic), \
        "Unable to calculate prod for dynamic shape {}.".format(shape)

    prod = 1
    for dim in shape:
        prod *= dim.get_min_length()
    return prod


def parse_input_value(input_value: str):
    """
    Parses a value of the --input command line parameter and gets a node name, shape and value.
    The node name includes a port if it is specified.
    Shape and value is equal to None if they are not specified.
    Parameters
    ----------
    input_value
        string with a specified node name, shape, value and data_type.
        E.g. 'node_name:0[4]{fp32}->[1.0 2.0 3.0 4.0]'

    Returns
    -------
        Node name, shape, value, data type
        E.g. 'node_name:0', '4', [1.0 2.0 3.0 4.0], np.float32
    """
    data_type = get_data_type_from_input_value(input_value)
    node_name = get_node_name_with_port_from_input_value(input_value)
    value = get_value_from_input_value(input_value)
    shape = get_shape_from_input_value(input_value)
    value_size = np.prod(len(value)) if isinstance(value, list) else 1

    if value is not None and shape is not None:
        for dim in shape:
            if isinstance(dim, Dimension) and dim.is_dynamic:
                raise Error("Cannot freeze input with dynamic shape: {}".format(shape))

    if shape is not None and value is not None and partial_shape_prod(shape) != value_size:
        raise Error("The shape '{}' of the input node '{}' does not correspond to the number of elements '{}' in the "
                    "value: {}".format(shape, node_name, value_size, value))
    return node_name, shape, value, data_type


def split_str_avoiding_square_brackets(s: str) -> list:
    """
    Splits a string by comma, but skips commas inside square brackets.
    :param s: string to split
    :return: list of strings split by comma
    """
    res = list()
    skipping = 0
    last_idx = 0
    for i, c in enumerate(s):
        if c == '[':
            skipping += 1
        elif c == ']':
            skipping -= 1
        elif c == ',' and skipping == 0:
            res.append(s[last_idx:i])
            last_idx = i + 1
    res.append(s[last_idx:])
    return res


def split_layouts_by_arrow(s: str) -> tuple:
    """
    Splits a layout string by first arrow (->).
    :param s: string to split
    :return: tuple containing source and target layouts
    """
    arrow = s.find('->')
    if arrow != -1:
        source_layout = s[:arrow]
        target_layout = s[arrow + 2:]
        if source_layout == '':
            source_layout = None
        if target_layout == '':
            target_layout = None
        return source_layout, target_layout
    else:
        return s, None


def validate_layout(layout: str):
    """
    Checks if layout is of valid format.
    :param layout: string containing layout
    :raises: if layout is incorrect
    """
    error_msg = 'Invalid layout parsed: {}'.format(layout)
    if layout:
        incorrect_brackets = xor(layout[0] == '[', layout[-1] == ']')
        if incorrect_brackets or layout[-1] == '-':
            error_msg += ', did you forget quotes?'
        else:
            valid_layout_re = re.compile(r'\[?[^\[\]\(\)\-\s]*\]?')
            if valid_layout_re.fullmatch(layout):
                return
        raise Error(error_msg)


def write_found_layout(name: str, found_layout: str, parsed: dict, dest: str = None):
    """
    Writes found layout data to the 'parsed' dict.
    :param name: name of the node to add layout
    :param found_layout: string containing layout for the node
    :param parsed: dict where result will be stored
    :param dest: type of the command line:
      * 'source' is --source_layout
      * 'target' is --target_layout
      * None is --layout
    """
    s_layout = None
    t_layout = None
    if name in parsed:
        s_layout = parsed[name]['source_layout']
        t_layout = parsed[name]['target_layout']
    if dest == 'source':
        s_layout = found_layout
    elif dest == 'target':
        t_layout = found_layout
    else:
        s_layout, t_layout = split_layouts_by_arrow(found_layout)
    validate_layout(s_layout)
    validate_layout(t_layout)
    parsed[name] = {'source_layout': s_layout, 'target_layout': t_layout}


def write_found_layout_list(idx: int, found_layout: str, parsed: list, dest: str = None):
    """
    Writes found layout data to the 'parsed' dict.
    :param idx: idx of of the node to add layout
    :param found_layout: string containing layout for the node
    :param parsed: list where result will be stored
    :param dest: type of the command line:
      * 'source' is --source_layout
      * 'target' is --target_layout
      * None is --layout
    """
    s_layout = None
    t_layout = None
    if idx < len(parsed):
        s_layout = parsed[idx]['source_layout']
        t_layout = parsed[idx]['target_layout']
    if dest == 'source':
        s_layout = found_layout
    elif dest == 'target':
        t_layout = found_layout
    else:
        s_layout, t_layout = split_layouts_by_arrow(found_layout)
    validate_layout(s_layout)
    validate_layout(t_layout)

    if idx < len(parsed):
        parsed[idx] = {'source_layout': s_layout, 'target_layout': t_layout}
    else:
        parsed.append({'source_layout': s_layout, 'target_layout': t_layout})


def parse_layouts_by_destination(s: str, parsed: dict, parsed_list: list, dest: str = None) -> None:
    """
    Parses layout command line to get all names and layouts from it. Adds all found data in the 'parsed' dict.
    :param s: string to parse
    :param parsed: dict where result will be stored
    :param dest: type of the command line:
      * 'source' is --source_layout
      * 'target' is --target_layout
      * None is --layout
    """
    list_s = split_str_avoiding_square_brackets(s)
    if len(list_s) == 1 and (list_s[0][-1] not in ')]' or (list_s[0][0] == '[' and list_s[0][-1] == ']')):
        # single layout case
        write_found_layout('', list_s[0], parsed, dest)
    else:
        for idx, layout_str in enumerate(list_s):
            # case for: "name1(nhwc->[n,c,h,w])"
            p1 = re.compile(r'([\w.:/\\]*)\((\S+)\)')
            m1 = p1.match(layout_str)
            # case for: "name1[n,h,w,c]->[n,c,h,w]"
            p2 = re.compile(r'([\w.:/\\]*)(\[\S*\])')
            m2 = p2.match(layout_str)
            if m1:
                found_g = m1.groups()
            elif m2:
                found_g = m2.groups()
            else:
                # case for layout without name
                write_found_layout_list(idx, layout_str, parsed_list, dest)
                continue
            if len(found_g[0]) > 0:
                write_found_layout(found_g[0], found_g[1], parsed, dest)
            else:
                write_found_layout_list(idx, found_g[1], parsed_list, dest)


def get_layout_values(argv_layout: str = '', argv_source_layout: str = '', argv_target_layout: str = ''):
    """
    Parses layout string.
    :param argv_layout: string with a list of layouts passed as a --layout.
    :param argv_source_layout: string with a list of layouts passed as a --source_layout.
    :param argv_target_layout: string with a list of layouts passed as a --target_layout.
    :return: dict with names and layouts associated
    """
    if argv_layout and (argv_source_layout or argv_target_layout):
        raise Error("--layout is used as well as --source_layout and/or --target_layout which is not allowed, please "
                    "use one of them.")
    res = {}
    res_list = []
    if argv_layout:
        parse_layouts_by_destination(argv_layout, res, res_list)
    if argv_source_layout:
        parse_layouts_by_destination(argv_source_layout, res, res_list, 'source')
    if argv_target_layout:
        parse_layouts_by_destination(argv_target_layout, res, res_list, 'target')
    if len(res) > 0 and len(res_list) > 0:
        raise Error("Some layout values are provided with names, and some without names. "
                    "Please provide ether all layouts with names or all layouts without names.")
    if len(res) > 0:
        return res
    else:
        return res_list


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
        for input_value in split_inputs(argv_input):
            node_name, _, value, _ = parse_input_value(input_value)
            input_node_names = input_node_names + ',' + node_name  if input_node_names != '' else node_name
            if value is None: # no value is specified for freezing
                continue
            if node_name in placeholder_values and placeholder_values[node_name] != value:
                raise Error("Overriding replacement value of the placeholder with name '{}': old value = {}, new value = {}"
                            ".".format(node_name, placeholder_values[node_name], value))
            placeholder_values[node_name] = value

    return placeholder_values, input_node_names


def split_inputs(input_str):
    brakets_count = 0
    inputs = []
    while input_str:
        idx = 0
        for c in input_str:
            if c == '[':
                brakets_count += 1
            if c == ']':
                brakets_count -= 1
            if c == ',':
                if brakets_count != 0:
                    idx += 1
                    continue
                else:
                    break
            idx += 1
        if idx >= len(input_str)-1:
            inputs.append(input_str)
            break
        inputs.append(input_str[:idx])
        input_str = input_str[idx+1:]
    return inputs



def split_shapes(argv_input_shape: str):
    range_reg = r'([0-9]*\.\.[0-9]*)'
    first_digit_reg = r'([0-9 ]+|-1|\?|{})'.format(range_reg)
    next_digits_reg = r'(,{})*'.format(first_digit_reg)
    tuple_reg = r'((\({}{}\))|(\[{}{}\]))'.format(first_digit_reg, next_digits_reg,
                                                  first_digit_reg, next_digits_reg)

    full_reg = r'^{}(\s*,\s*{})*$|^$'.format(tuple_reg, tuple_reg)
    if not re.match(full_reg, argv_input_shape):
        raise Error('Input shape "{}" cannot be parsed. ' + refer_to_faq_msg(57), argv_input_shape)
    return re.findall(r'[(\[]([0-9,\.\? -]+)[)\]]', argv_input_shape)

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
        E.g. '[1,2],[3,4]'.
        Only positive integers are accepted.
        '?' marks dynamic dimension.
        Partial shape is specified with ellipsis. E.g. '[1..10,2,3]'
    argv_batch
        integer that overrides batch size in input shape

    Returns
    -------
        parsed shapes in form of {'name of input':tuple} if names of inputs are provided with shapes
        parsed shapes in form of {'name of input':None} if names of inputs are provided without shapes
        tuple if only one shape is provided and no input name
        None if neither shape nor input were provided
    """
    if argv_input_shape and argv_batch:
        raise Error("Both --input_shape and --batch were provided. Please provide only one of them. " +
                    refer_to_faq_msg(56))

    # attempt to extract shapes from --input parameters
    placeholder_shapes = dict()
    placeholder_data_types = dict()
    are_shapes_specified_through_input = False
    inputs_list = list()
    if argv_input:
        for input_value in split_inputs(argv_input):
            node_name, shape, _, data_type = parse_input_value(input_value)
            placeholder_shapes[node_name] = shape
            inputs_list.append(node_name)
            if data_type is not None:
                placeholder_data_types[node_name] = data_type
            if shape is not None:
                are_shapes_specified_through_input = True

    if argv_input_shape and are_shapes_specified_through_input:
        raise Error("Shapes are specified using both --input and --input_shape command-line parameters, but only one "
                    "parameter is allowed.")

    if argv_batch and are_shapes_specified_through_input:
        raise Error("Shapes are specified using both --input and --batch command-line parameters, but only one "
                    "parameter is allowed.")

    if are_shapes_specified_through_input:
        return inputs_list, placeholder_shapes, placeholder_data_types

    shapes = list()
    inputs = list()
    inputs_list = list()
    placeholder_shapes = None


    if argv_input_shape:
        shapes = split_shapes(argv_input_shape)

    if argv_input:
        inputs = split_inputs(argv_input)
    inputs = [remove_data_type_from_input_value(inp) for inp in inputs]

    # check number of shapes with no input provided
    if argv_input_shape and not argv_input:
        placeholder_shapes = [PartialShape(shape) for shape in shapes]
        if len(placeholder_shapes) == 1:
            placeholder_shapes = PartialShape(placeholder_shapes[0])
    # check if number of shapes does not match number of passed inputs
    elif argv_input and (len(shapes) == len(inputs) or len(shapes) == 0):
        # clean inputs from values for freezing
        inputs_without_value = list(map(lambda x: x.split('->')[0], inputs))
        placeholder_shapes = dict(zip_longest(inputs_without_value,
                                              map(lambda x: PartialShape(x) if x else None, shapes)))
        for inp in inputs:
            if '->' not in inp:
                inputs_list.append(inp)
                continue
            shape = placeholder_shapes[inp.split('->')[0]]
            inputs_list.append(inp.split('->')[0])

            if shape is None:
                continue
            for dim in shape:
                if isinstance(dim, Dimension) and not dim.is_static:
                    raise Error("Cannot freeze input with dynamic shape: {}".format(shape))

    elif argv_input:
        raise Error('Please provide each input layers with an input layer shape. ' + refer_to_faq_msg(58))

    return inputs_list, placeholder_shapes, placeholder_data_types


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
        "(10,20,30),(11,22,33)" -> [mo_array(10,20,30), mo_array(11,22,33)]
    """
    res = {}
    if not argv_values:
        return res

    matches = [m for m in re.finditer(r'[(\[]([0-9., -]+)[)\]]', argv_values, re.IGNORECASE)]

    error_msg = 'Mean/scale values should consist of name and values specified in round or square brackets ' \
                'separated by comma, e.g. data(1,2,3),info[2,3,4],egg[255] or data(1,2,3). Or just plain set of ' \
                'values without names: (1,2,3),(2,3,4) or [1,2,3],[2,3,4].' + refer_to_faq_msg(101)
    if not matches:
        raise Error(error_msg, argv_values)

    name_start_idx = 0
    name_was_present = False
    for idx, match in enumerate(matches):
        input_name = argv_values[name_start_idx:match.start(0)]
        name_start_idx = match.end(0) + 1
        tuple_value = np.fromstring(match.groups()[0], dtype=float, sep=',')

        if idx != 0 and (name_was_present ^ bool(input_name)):
            # if node name firstly was specified and then subsequently not or vice versa
            # e.g. (255),input[127] or input(255),[127]
            raise Error(error_msg, argv_values)

        name_was_present = True if input_name != "" else False
        if name_was_present:
            res[input_name] = tuple_value
        else:
            res[idx] = tuple_value

    if not name_was_present:
        # return a list instead of a dictionary
        res = sorted(res.values(), key=lambda v: v[0])
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
    mean = { 'data': np.array, 'info': np.array }, scale = { 'data': np.array, 'info': np.array }, input = "data, info" ->
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

        for inp_scale in scale_values.keys():
            if inp_scale not in inputs:
                raise Error("Specified scale_values name '{}' do not match to any of inputs: {}. "
                            "Please set 'scale_values' that correspond to values from input.".format(inp_scale, inputs))

        for inp_mean in mean_values.keys():
            if inp_mean not in inputs:
                raise Error("Specified mean_values name '{}' do not match to any of inputs: {}. "
                            "Please set 'mean_values' that correspond to values from input.".format(inp_mean, inputs))

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
    # mean and/or scale are specified without inputs
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


def get_model_name_from_args(argv: argparse.Namespace):
    model_name = "<UNKNOWN_NAME>"
    if hasattr(argv, 'model_name'):
        if argv.model_name:
            model_name = argv.model_name
        elif argv.input_model:
            model_name = get_model_name(argv.input_model)
        elif argv.saved_model_dir:
            model_name = "saved_model"
        elif argv.input_meta_graph:
            model_name = get_model_name(argv.input_meta_graph)
        elif argv.input_symbol:
            model_name = get_model_name(argv.input_symbol)
        argv.model_name = model_name
    return model_name


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


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isbool(value):
    try:
        strtobool(value)
        return True
    except ValueError:
        return False


def isdict(value):
    try:
        evaluated = ast.literal_eval(value)
        return isinstance(evaluated, dict)
    except ValueError:
        return False


def convert_string_to_real_type(value: str):
    if isdict(value):
        return ast.literal_eval(value)

    values = value.split(',')
    for i in range(len(values)):
        value = values[i]
        if value.isdigit():
            values[i] = int(value)
        elif isfloat(value):
            values[i] = float(value)
        elif isbool(value):
            values[i] = strtobool(value)

    return values[0] if len(values) == 1 else values


def parse_transform(transform: str) -> list:
    transforms = []

    if len(transform) == 0:
        return transforms

    all_transforms = re.findall(r"([a-zA-Z0-9]+)(\[([^\]]+)\])*(,|$)", transform)

    # Check that all characters were matched otherwise transform key value is invalid
    key_len = len(transform)
    for transform in all_transforms:
        # In regexp we have 4 groups where 1st group - transformation_name,
        #                                  2nd group - [args],
        #                                  3rd group - args, <-- nested group
        #                                  4th group - EOL
        # And to check that regexp matched all string we decrease total length by the length of matched groups (1,2,4)
        # In case if no arguments were given to transformation then 2nd and 3rd groups will be empty.
        if len(transform) != 4:
            raise Error("Unexpected transform key structure: {}".format(transform))
        key_len -= len(transform[0]) + len(transform[1]) + len(transform[3])

    if key_len != 0:
        raise Error("Unexpected transform key structure: {}".format(transform))

    for transform in all_transforms:
        name = transform[0]
        args = transform[2]

        args_dict = {}

        if len(args) != 0:
            for arg in args.split(';'):
                m = re.match(r"^([_a-zA-Z]+)=(.+)$", arg)
                if not m:
                    raise Error("Unrecognized attributes for transform key: {}".format(transform))

                args_dict[m.group(1)] = convert_string_to_real_type(m.group(2))

        transforms.append((name, args_dict))

    return transforms


def check_available_transforms(transforms: list):
    """
    This function check that transformations specified by user are available.
    :param transforms: list of user specified transformations
    :return: raises an Error if transformation is not available
    """
    from openvino.tools.mo.back.offline_transformations import get_available_transformations
    available_transforms = get_available_transformations()

    missing_transformations = []
    for name, _ in transforms:
        if name not in available_transforms.keys():
            missing_transformations.append(name)

    if len(missing_transformations) != 0:
        raise Error('Following transformations ({}) are not available. '
                    'List with available transformations ({})'.format(','.join(missing_transformations),
                                                                      ','.join(available_transforms.keys())))
    return True


def check_positive(value):
    try:
        int_value = int(value)
        if int_value <= 0:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("expected a positive integer value")

    return int_value


def check_bool(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        if value.lower() not in ['true', 'false']:
            raise argparse.ArgumentTypeError("expected a True/False value")
        return value.lower() == 'true'
    else:
        raise argparse.ArgumentTypeError("expected a bool or str type")


def depersonalize(value: str, key: str):
    dir_keys = [
        'output_dir', 'extensions', 'saved_model_dir', 'tensorboard_logdir', 'caffe_parser_path'
    ]

    if isinstance(value, list):
        updated_value = []
        for elem in value:
            updated_value.append(depersonalize(elem, key))
        return updated_value

    if not isinstance(value, str):
        return value
    res = []
    for path in value.split(','):
        if os.path.isdir(path) and key in dir_keys:
            res.append('DIR')
        elif os.path.isfile(path):
            res.append(os.path.join('DIR', os.path.split(path)[1]))
        else:
            res.append(path)
    return ','.join(res)

def get_available_front_ends(fem=None):
    # Use this function as workaround to avoid IR frontend usage by MO
    if fem is None:
        return []
    available_moc_front_ends = fem.get_available_front_ends()
    if 'ir' in available_moc_front_ends:
        available_moc_front_ends.remove('ir')

    return available_moc_front_ends
