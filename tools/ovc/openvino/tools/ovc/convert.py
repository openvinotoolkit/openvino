# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
from collections import namedtuple
from typing import Any

from openvino.runtime import PartialShape, Shape, Layout, Model # pylint: disable=no-name-in-module,import-error

from openvino.tools.ovc.convert_impl import _convert
from openvino.tools.ovc.logger import get_logger_state, restore_logger_state
from openvino.tools.ovc.cli_parser import get_all_cli_parser

InputCutInfo = namedtuple("InputInfo", ["name", "shape", "type", "value"], defaults=[None, None, None, None])
LayoutMap = namedtuple("LayoutMap", ["source_layout", "target_layout"], defaults=[None, None])


def convert_model(
        input_model: [str, pathlib.Path, Any, list],    # TODO: Instead of list just accept arbitrary number of positional arguments

        # Framework-agnostic parameters
        input: [str, list, tuple, InputCutInfo] = None,
        output: [str, list] = None,
        example_input: Any = None,
        source_layout: [str, Layout, dict] = (),
        target_layout: [str, Layout, dict] = (),
        layout: [str, Layout, LayoutMap, list, dict] = (),
        extensions: [str, pathlib.Path, list, Any] = None,
        transform: [str, list, tuple] = "",  #TODO: Consider removing
        silent: bool = True,    #TODO: Consider removing
        log_level: str = 'ERROR',

        # PaddlePaddle-specific parameters:
        example_output: Any = None,  # TODO: Consider removing

        # TensorFlow*-specific parameters
        saved_model_tags: [str, list] = None,   # TODO: Consider removing
) -> Model:
    """
    Converts the model from original framework to OpenVino Model.

    Framework-agnostic parameters:
        :param input_model:
            Model object in original framework (PyTorch, Tensorflow) or path to model file.
            Tensorflow*: a file with a pre-trained model (binary or text .pb file after freezing).
            Caffe*: a model proto file with model weights

            Supported formats of input model:

            PaddlePaddle
            paddle.hapi.model.Model
            paddle.fluid.dygraph.layers.Layer
            paddle.fluid.executor.Executor

            PyTorch
            torch.nn.Module
            torch.jit.ScriptModule
            torch.jit.ScriptFunction

            TF
            tf.compat.v1.Graph
            tf.compat.v1.GraphDef
            tf.compat.v1.wrap_function
            tf.compat.v1.session

            TF2 / Keras
            tf.keras.Model
            tf.keras.layers.Layer
            tf.function
            tf.Module
            tf.train.checkpoint

        :param input:
            Input can be set by passing a list of InputCutInfo objects or by a list
            of tuples. Each tuple can contain optionally input name, input
            type or input shape. Example: input=("op_name", PartialShape([-1,
            3, 100, 100]), Type(np.float32)). Alternatively input can be set by
            a string or list of strings of the following format. Quoted list of comma-separated
            input nodes names with shapes, data types, and values for freezing.
            If operation names are specified, the order of inputs in converted
            model will be the same as order of specified operation names (applicable for TF2, ONNX, MxNet).
            The shape and value are specified as comma-separated lists. The data type of input node is specified
            in braces and can have one of the values: f64 (float64), f32 (float32), f16 (float16), i64
            (int64), i32 (int32), u8 (uint8), boolean (bool). Data type is optional.
            If it's not specified explicitly then there are two options: if input
            node is a parameter, data type is taken from the original node dtype,
            if input node is not a parameter, data type is set to f32. Example, to set
            `input_1` with shape [1,100], and Parameter node `sequence_len` with
            scalar input with value `150`, and boolean input `is_training` with
            `False` value use the following format: "input_1[1,100],sequence_len->150,is_training->False".
            Another example, use the following format to set input port 0 of the node
            `node_name1` with the shape [3,4] as an input node and freeze output
            port 1 of the node `node_name2` with the value [20,15] of the int32 type
            and shape [2]: "0:node_name1[3,4],node_name2:1[2]{i32}->[20,15]".
        :param output:
            The name of the output operation of the model or list of names. For TensorFlow*,
            do not add :0 to this name.The order of outputs in converted model is the
            same as order of specified operation names.
        :param example_input:
            Sample of model input in original framework.
            For PyTorch it can be torch.Tensor.
            For Tensorflow it can be tf.Tensor or numpy.ndarray.
            For PaddlePaddle it can be Paddle Variable.
        :param source_layout:
            Layout of the input or output of the model in the framework. Layout can
            be set by passing a dictionary, where key is input name and value is LayoutMap
            object. Or layout can be set by string of the following format. Layout
            can be specified in the short form, e.g. nhwc, or in complex form, e.g.
            "[n,h,w,c]". Example for many names: "in_name1([n,h,w,c]),in_name2(nc),out_name1(n),out_name2(nc)".
            Layout can be partially defined, "?" can be used to specify undefined
            layout for one dimension, "..." can be used to specify undefined layout
            for multiple dimensions, for example "?c??", "nc...", "n...c", etc.
        :param target_layout:
            Same as "source_layout", but specifies target layout that will be in
            the model after processing by ModelOptimizer.
        :param layout:
            Combination of "source_layout" and "target_layout". Can't be used
            with either of them. If model has one input it is sufficient to specify
            layout of this input, for example "layout" nhwc. To specify layouts
            of many tensors, names must be provided, for example: layout="name1(nchw),name2(nc)".
            It is possible to instruct ModelOptimizer to change layout, for example:
                layout="name1(nhwc->nchw),name2(cn->nc)".
            Also "*" in long layout form can be used to fuse dimensions, for example "[n,c,...]->[n*c,...]".
        :param extensions:
            Paths to libraries (.so or .dll) with extensions, comma-separated
            list of paths, objects derived from BaseExtension class or lists of
            objects. For the legacy MO path (if "use_legacy_frontend" is used),
            a directory or a comma-separated list of directories with extensions
            are supported. To disable all extensions including those that are placed
            at the default location, pass an empty string.
        :param transform:
            Apply additional transformations. 'transform' can be set by a list
            of tuples, where the first element is transform name and the second element
            is transform parameters. For example: [('LowLatency2', {{'use_const_initializer':
            False}}), ...] transform="transformation_name1[args],transformation_name2..."
            where [args] is key=value pairs separated by semicolon. Examples:
                     transform="LowLatency2" or
                     transform="Pruning" or
                     transform="LowLatency2[use_const_initializer=False]" or
                     transform="MakeStateful[param_res_names=
                     {'input_name_1':'output_name_1','input_name_2':'output_name_2'}]"
            Available transformations: "LowLatency2", "MakeStateful", "Pruning"
        :param silent:
            Prevent any output messages except those that correspond to log level
            equals ERROR, that can be set with the following option: "log_level".
            By default, log level is already ERROR.
        :param log_level:
            Logger level of logging massages from MO.
            Expected one of ['CRITICAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'].

    PaddlePaddle-specific parameters:
        :param example_output:
            Sample of model output in original framework. For PaddlePaddle it can be Paddle Variable.

    TensorFlow*-specific parameters:
        :param saved_model_tags:
            Group of tag(s) of the MetaGraphDef to load, in string format, separated
            by ','. For tag-set contains multiple tags, all tags must be passed in.

    Returns:
        openvino.runtime.Model
    """
    params = locals()
    logger_state = get_logger_state()
    cli_parser = get_all_cli_parser()
    ov_model, _ = _convert(cli_parser, params, True)
    restore_logger_state(logger_state)
    return ov_model
