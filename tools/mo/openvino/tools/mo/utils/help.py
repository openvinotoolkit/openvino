# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def get_convert_model_help_specifics():
    from openvino.tools.mo.utils.cli_parser import CanonicalizeTransformationPathCheckExistenceAction, \
        CanonicalizePathCheckExistenceAction, CanonicalizeExtensionsPathCheckExistenceAction, \
        CanonicalizePathCheckExistenceIfNeededAction, readable_file_or_dir, readable_dirs_or_files_or_empty, \
        check_positive
    from openvino.tools.mo.utils.version import VersionChecker
    return {
        'input_model':
            {'description':
                 'Tensorflow*: a file with a pre-trained model '
                 '(binary or text .pb file after freezing). '
                 'Caffe*: a model proto file with model weights.', 'action': CanonicalizePathCheckExistenceAction,
             'type': readable_file_or_dir,
             'aliases': {'-w', '-m'}},
        'input_shape':
            {'description':
                 'Input shape(s) that should be fed to an input node(s) '
                 'of the model. Shape is defined as a comma-separated '
                 'list of integer numbers enclosed in parentheses or '
                 'square brackets, for example [1,3,227,227] or '
                 '(1,227,227,3), where the order of dimensions depends '
                 'on the framework input layout of the model. For '
                 'example, [N,C,H,W] is used for ONNX* models and '
                 '[N,H,W,C] for TensorFlow* models. The shape can '
                 'contain undefined dimensions (? or -1) and should fit '
                 'the dimensions defined in the input operation of the '
                 'graph. Boundaries of undefined dimension can be '
                 'specified with ellipsis, for example '
                 '[1,1..10,128,128]. One boundary can be undefined, for '
                 'example [1,..100] or [1,3,1..,1..]. If there are '
                 'multiple inputs in the model, --input_shape should '
                 'contain definition of shape for each input separated '
                 'by a comma, for example: [1,3,227,227],[2,4] for a '
                 'model with two inputs with 4D and 2D shapes. '
                 'Alternatively, specify shapes with the --input option.'},
        'input':
            {'description':
                 'Quoted list of comma-separated input nodes names with '
                 'shapes, data types, and values for freezing. The order '
                 'of inputs in converted model is the same as order of '
                 'specified operation names. The shape and value are '
                 'specified as comma-separated lists. The data type of '
                 'input node is specified in braces and can have one of '
                 'the values: f64 (float64), f32 (float32), f16 '
                 '(float16), i64 (int64), i32 (int32), u8 (uint8), '
                 'boolean (bool). Data type is optional. If it\'s not '
                 'specified explicitly then there are two options: if '
                 'input node is a parameter, data type is taken from the '
                 'original node dtype, if input node is not a parameter, '
                 'data type is set to f32. Example, to set `input_1` '
                 'with shape [1,100], and Parameter node `sequence_len` '
                 'with scalar input with value `150`, and boolean input '
                 '`is_training` with `False` value use the following '
                 'format: \n '
                 '\"input_1[1,100],sequence_len->150,is_training->False\". '
                 'Another example, use the following format to set input '
                 'port 0 of the node `node_name1` with the shape [3,4] '
                 'as an input node and freeze output port 1 of the node '
                 '\"node_name2\" with the value [20,15] of the int32 type '
                 'and shape [2]: \n '
                 '\"0:node_name1[3,4],node_name2:1[2]{i32}->[20,15]\".'},
        'mean_values':
            {'description':
                 'Mean values to be used for the input image per '
                 'channel. Values to be provided in the (R,G,B) or '
                 '[R,G,B] format. Can be defined for desired input of '
                 'the model, for example: "--mean_values '
                 'data[255,255,255],info[255,255,255]". The exact '
                 'meaning and order of channels depend on how the '
                 'original model was trained.'},
        'scale_values':
            {'description':
                 'Scale values to be used for the input image per '
                 'channel. Values are provided in the (R,G,B) or [R,G,B] '
                 'format. Can be defined for desired input of the model, '
                 'for example: "--scale_values '
                 'data[255,255,255],info[255,255,255]". The exact '
                 'meaning and order of channels depend on how the '
                 'original model was trained. If both --mean_values and '
                 '--scale_values are specified, the mean is subtracted '
                 'first and then scale is applied regardless of the '
                 'order of options in command line.'},
        'source_layout':
            {'description':
                 'Layout of the input or output of the model in the '
                 'framework. Layout can be specified in the short form, '
                 'e.g. nhwc, or in complex form, e.g. \"[n,h,w,c]\". '
                 'Example for many names: \"in_name1([n,h,w,c]),in_name2('
                 'nc),out_name1(n),out_name2(nc)\". Layout can be '
                 'partially defined, \"?\" can be used to specify '
                 'undefined layout for one dimension, \"...\" can be used '
                 'to specify undefined layout for multiple dimensions, '
                 'for example \"?c??\", \"nc...\", \"n...c\", etc.'},
        'transform':
            {'description':
                 'Apply additional transformations. Usage: \"--transform '
                 'transformation_name1[args],transformation_name2...\" '
                 'where [args] is key=value pairs separated by '
                 'semicolon. Examples: \"--transform LowLatency2\" or \"--'
                 'transform Pruning" or "--transform '
                 'LowLatency2[use_const_initializer=False]" or "--'
                 'transform "MakeStateful[param_res_names= {\'input_name_1\':'
                 '\'output_name_1\',\'input_name_2\':\'output_name_2\'}]\" \n'
                 'Available transformations: "LowLatency2", "MakeStateful", "Pruning"'},
        'extensions':
            {'description':
                 'Paths or a comma-separated list of paths to libraries '
                 '(.so or .dll) with extensions. For the legacy MO path '
                 '(if `--use_legacy_frontend` is used), a directory or a '
                 'comma-separated list of directories with extensions '
                 'are supported. To disable all extensions including '
                 'those that are placed at the default location, pass an empty string.',
             'action': CanonicalizeExtensionsPathCheckExistenceAction,
             'type': readable_dirs_or_files_or_empty},
        'transformations_config':
            {'description':
                 'Use the configuration file with transformations '
                 'description. Transformations file can be specified as '
                 'relative path from the current directory, as absolute '
                 'path or as arelative path from the mo root directory.',
             'action': CanonicalizeTransformationPathCheckExistenceAction},
        'counts':
            {'action': CanonicalizePathCheckExistenceIfNeededAction},
        'version':
            {'action': 'version',
             'version': 'Version of Model Optimizer is: {}'.format(VersionChecker().get_ie_version())},
        'scale':
            {'type': float,
             'aliases': {'-s'}},
        'batch':
            {'type': check_positive,
             'aliases': {'-b'}},
        'input_proto':
            {'aliases': {'-d'}},
        'log_level':
            {'choices': ['CRITICAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']}
    }


# TODO: remove this when internal converting of params to string is removed
def get_to_string_methods_for_params():
    from openvino.tools.mo.utils.cli_parser import path_to_str_or_object, str_list_to_str, \
        mean_scale_value_to_str, source_target_layout_to_str, layout_param_to_str, transform_param_to_str, \
        extensions_to_str_or_extensions_class, batch_to_int, transformations_config_to_str
    return {
        'input_model': path_to_str_or_object,
        'output': str_list_to_str,
        'mean_values': mean_scale_value_to_str,
        'scale_values': mean_scale_value_to_str,
        'source_layout': source_target_layout_to_str,
        'target_layout': source_target_layout_to_str,
        'layout': layout_param_to_str,
        'transform': transform_param_to_str,
        'extensions': extensions_to_str_or_extensions_class,
        'batch': batch_to_int,
        'transformations_config': transformations_config_to_str,
        'saved_model_tags': str_list_to_str
    }
