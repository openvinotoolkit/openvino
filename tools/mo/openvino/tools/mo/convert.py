# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
from collections import namedtuple
from typing import Any

from openvino.runtime import PartialShape, Shape, Layout, Model
from openvino.tools.mo.convert_impl import _convert
from openvino.tools.mo.utils.cli_parser import get_all_cli_parser  # pylint: disable=no-name-in-module,import-error
from openvino.tools.mo.utils.logger import get_logger_state, restore_logger_state  # pylint: disable=no-name-in-module,import-error

LayoutMap = namedtuple("LayoutMap", ["source_layout", "target_layout"], defaults=[None, None])
InputCutInfo = namedtuple("InputInfo", ["name", "shape", "type", "value"], defaults=[None, None, None, None])


def convert_model(
        input_model: [str, pathlib.Path, Any] = None,

        # Optional parameters
        help: bool = False,
        framework: [str] = None,

        # Framework-agnostic parameters
        input: [str, list, tuple, InputCutInfo] = None,
        output: [str, list] = None,
        input_shape: [str, PartialShape, Shape, list] = None,
        example_input: Any = None,
        batch: int = None,
        mean_values: [str, dict, list] = (),
        scale_values: [str, dict, list] = (),
        scale: [str, float] = None,
        reverse_input_channels: bool = False,
        source_layout: [str, Layout, dict] = (),
        target_layout: [str, Layout, dict] = (),
        layout: [str, Layout, LayoutMap, list, dict] = (),
        compress_to_fp16: bool = False,
        extensions: [str, pathlib.Path, list, Any] = None,
        transform: [str, list, tuple] = "",
        transformations_config: [str, pathlib.Path] = None,
        silent: bool = True,
        log_level: str = 'ERROR',
        version: bool = None,
        progress: bool = False,
        stream_output: bool = False,
        share_weights: bool = False,

        # PaddlePaddle-specific parameters:
        example_output: Any = None,

        # TensorFlow*-specific parameters
        input_model_is_text: bool = None,
        input_checkpoint: [str, pathlib.Path] = None,
        input_meta_graph: [str, pathlib.Path] = None,
        saved_model_dir: [str, pathlib.Path] = None,
        saved_model_tags: [str, list] = None,
        tensorflow_custom_operations_config_update: [str, pathlib.Path] = None,
        tensorflow_object_detection_api_pipeline_config: [str, pathlib.Path] = None,
        tensorboard_logdir: [str, pathlib.Path] = None,
        tensorflow_custom_layer_libraries: [str, pathlib.Path] = None,

        # Caffe*-specific parameters:
        input_proto: [str, pathlib.Path] = None,
        caffe_parser_path: [str, pathlib.Path] = None,
        k: [str, pathlib.Path] = None,
        disable_omitting_optional: bool = False,
        enable_flattening_nested_params: bool = False,

        # Kaldi-specific parameters:
        counts: [str, pathlib.Path] = None,
        remove_output_softmax: bool = False,
        remove_memory: bool = False,

        **args
) -> Model:
    """
    Converts the model from original framework to OpenVino Model.

    Args:
        :param help:
            Print available parameters.
        :param framework:
            Name of the framework used to train the input model.

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
            model will be the same as order of specified operation names (applicable for TF2, ONNX).
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
        :param input_shape:
            Input shape(s) that should be fed to an input node(s) of the model. Input
            shapes can be defined by passing a list of objects of type PartialShape,
            Shape, [Dimension, ...] or [int, ...] or by a string of the following
            format. Shape is defined as a comma-separated list of integer numbers
            enclosed in parentheses or square brackets, for example [1,3,227,227]
            or (1,227,227,3), where the order of dimensions depends on the framework
            input layout of the model. For example, [N,C,H,W] is used for ONNX* models
            and [N,H,W,C] for TensorFlow* models. The shape can contain undefined
            dimensions (? or -1) and should fit the dimensions defined in the input
            operation of the graph. Boundaries of undefined dimension can be specified
            with ellipsis, for example [1,1..10,128,128]. One boundary can be
            undefined, for example [1,..100] or [1,3,1..,1..]. If there are multiple
            inputs in the model, --input_shape should contain definition of shape
            for each input separated by a comma, for example: [1,3,227,227],[2,4]
            for a model with two inputs with 4D and 2D shapes. Alternatively, specify
            shapes with the --input option.
        :param example_input:
            Sample of model input in original framework.
            For PyTorch it can be torch.Tensor.
            For Tensorflow it can be tf.Tensor or numpy.ndarray.
            For PaddlePaddle it can be Paddle Variable.
        :param batch:
            Set batch size. It applies to 1D or higher dimension inputs.
            The default dimension index for the batch is zero.
            Use a label 'n' in --layout or --source_layout option to set the batch dimension.
            For example, "x(hwnc)" defines the third dimension to be the batch.
        :param mean_values:
            Mean values to be used for the input image per channel. Mean values can
            be set by passing a dictionary, where key is input name and value is mean
            value. For example mean_values={'data':[255,255,255],'info':[255,255,255]}.
            Or mean values can be set by a string of the following format. Values to
            be provided in the (R,G,B) or [R,G,B] format. Can be defined for desired
            input of the model, for example: "--mean_values data[255,255,255],info[255,255,255]".
            The exact meaning and order of channels depend on how the original model
            was trained.
        :param scale_values:
            Scale values to be used for the input image per channel. Scale values
            can be set by passing a dictionary, where key is input name and value is
            scale value. For example scale_values={'data':[255,255,255],'info':[255,255,255]}.
            Or scale values can be set by a string of the following format. Values
            are provided in the (R,G,B) or [R,G,B] format. Can be defined for desired
            input of the model, for example: "--scale_values data[255,255,255],info[255,255,255]".
            The exact meaning and order of channels depend on how the original model
            was trained. If both --mean_values and --scale_values are specified,
            the mean is subtracted first and then scale is applied regardless of
            the order of options in command line.
        :param scale:
            All input values coming from original network inputs will be divided
            by this value. When a list of inputs is overridden by the --input parameter,
            this scale is not applied for any input that does not match with the original
            input of the model. If both --mean_values and --scale  are specified,
            the mean is subtracted first and then scale is applied regardless of
            the order of options in command line.
        :param reverse_input_channels:
            Switch the input channels order from RGB to BGR (or vice versa). Applied
            to original inputs of the model if and only if a number of channels equals
            3. When --mean_values/--scale_values are also specified, reversing
            of channels will be applied to user's input data first, so that numbers
            in --mean_values and --scale_values go in the order of channels used
            in the original model. In other words, if both options are specified,
            then the data flow in the model looks as following: Parameter -> ReverseInputChannels
            -> Mean apply-> Scale apply -> the original body of the model.
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
            Same as --source_layout, but specifies target layout that will be in
            the model after processing by ModelOptimizer.
        :param layout:
            Combination of --source_layout and --target_layout. Can't be used
            with either of them. If model has one input it is sufficient to specify
            layout of this input, for example --layout nhwc. To specify layouts
            of many tensors, names must be provided, for example: --layout "name1(nchw),name2(nc)".
            It is possible to instruct ModelOptimizer to change layout, for example:
                --layout "name1(nhwc->nchw),name2(cn->nc)".
            Also "*" in long layout form can be used to fuse dimensions, for example "[n,c,...]->[n*c,...]".
        :param compress_to_fp16:
            If the original model has FP32 weights or biases, they are compressed
            to FP16. All intermediate data is kept in original precision. Option
            can be specified alone as "--compress_to_fp16", or explicit True/False
            values can be set, for example: "--compress_to_fp16=False", or "--compress_to_fp16=True"
        :param extensions:
            Paths to libraries (.so or .dll) with extensions, comma-separated
            list of paths, objects derived from BaseExtension class or lists of
            objects. For the legacy MO path (if `--use_legacy_frontend` is used),
            a directory or a comma-separated list of directories with extensions
            are supported. To disable all extensions including those that are placed
            at the default location, pass an empty string.
        :param transform:
            Apply additional transformations. 'transform' can be set by a list
            of tuples, where the first element is transform name and the second element
            is transform parameters. For example: [('LowLatency2', {{'use_const_initializer':
            False}}), ...]"--transform transformation_name1[args],transformation_name2..."
            where [args] is key=value pairs separated by semicolon. Examples:
                     "--transform LowLatency2" or
                     "--transform Pruning" or
                     "--transform LowLatency2[use_const_initializer=False]" or
                     "--transform "MakeStateful[param_res_names=
            {'input_name_1':'output_name_1','input_name_2':'output_name_2'}]""
            Available transformations: "LowLatency2", "MakeStateful", "Pruning"
        :param transformations_config:
            Use the configuration file with transformations description or pass
            object derived from BaseExtension class. Transformations file can
            be specified as relative path from the current directory, as absolute
            path or as relative path from the mo root directory.
        :param silent:
            Prevent any output messages except those that correspond to log level
            equals ERROR, that can be set with the following option: --log_level.
            By default, log level is already ERROR.
        :param log_level:
            Logger level of logging massages from MO.
            Expected one of ['CRITICAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'].
        :param version:
            Version of Model Optimizer
        :param progress:
            Enable model conversion progress display.
        :param stream_output:
            Switch model conversion progress display to a multiline mode.
        :param share_weights:
            Map memory of weights instead reading files or share memory from input model.
            Currently, mapping feature is provided only for ONNX models
            that do not require fallback to the legacy ONNX frontend for the conversion.

    PaddlePaddle-specific parameters:
        :param example_output:
            Sample of model output in original framework. For PaddlePaddle it can be Paddle Variable.

    TensorFlow*-specific parameters:
        :param input_model_is_text:
            TensorFlow*: treat the input model file as a text protobuf format. If
            not specified, the Model Optimizer treats it as a binary file by default.
        :param input_checkpoint:
            TensorFlow*: variables file to load.
        :param input_meta_graph:
            Tensorflow*: a file with a meta-graph of the model before freezing
        :param saved_model_dir:
            TensorFlow*: directory with a model in SavedModel format of TensorFlow
            1.x or 2.x version.
        :param saved_model_tags:
            Group of tag(s) of the MetaGraphDef to load, in string format, separated
            by ','. For tag-set contains multiple tags, all tags must be passed in.
        :param tensorflow_custom_operations_config_update:
            TensorFlow*: update the configuration file with node name patterns
            with input/output nodes information.
        :param tensorflow_object_detection_api_pipeline_config:
            TensorFlow*: path to the pipeline configuration file used to generate
            model created with help of Object Detection API.
        :param tensorboard_logdir:
            TensorFlow*: dump the input graph to a given directory that should be
            used with TensorBoard.
        :param tensorflow_custom_layer_libraries:
            TensorFlow*: comma separated list of shared libraries with TensorFlow*
            custom operations implementation.

    Caffe*-specific parameters:
        :param input_proto:
            Deploy-ready prototxt file that contains a topology structure and
            layer attributes
        :param caffe_parser_path:
            Path to Python Caffe* parser generated from caffe.proto
        :param k:
            Path to CustomLayersMapping.xml to register custom layers
        :param disable_omitting_optional:
            Disable omitting optional attributes to be used for custom layers.
            Use this option if you want to transfer all attributes of a custom layer
            to IR. Default behavior is to transfer the attributes with default values
            and the attributes defined by the user to IR.
        :param enable_flattening_nested_params:
            Enable flattening optional params to be used for custom layers. Use
            this option if you want to transfer attributes of a custom layer to IR
            with flattened nested parameters. Default behavior is to transfer
            the attributes without flattening nested parameters.

    Kaldi-specific parameters:
        :param counts:
            Path to the counts file
        :param remove_output_softmax:
            Removes the SoftMax layer that is the output layer
        :param remove_memory:
            Removes the Memory layer and use additional inputs outputs instead

    Returns:
        openvino.runtime.Model
    """
    params = locals()
    logger_state = get_logger_state()
    del params['args']
    params.update(args)
    cli_parser = get_all_cli_parser()
    ov_model, _ = _convert(cli_parser, framework, params, True)
    restore_logger_state(logger_state)
    return ov_model
