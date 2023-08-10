# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def get_convert_model_help_specifics():
    from openvino.tools.ovc.cli_parser import CanonicalizePathCheckExistenceAction, readable_dirs_or_files_or_empty, readable_files_or_empty
    from openvino.tools.ovc.version import VersionChecker
    return {
        'input_model':
            {'description':
                 'Input model file(s) from TensorFlow, ONNX, PaddlePaddle. '
                 'Use openvino.convert_model in Python to convert models from Pytorch.'
                 '',
             'action': CanonicalizePathCheckExistenceAction,
             'type': readable_dirs_or_files_or_empty,
             'aliases': {}},
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
                 'Information for model input required for model conversion. '
                 'This is a comma separated list with optional '
                 'input names, shapes, data types, and values for freezing. '
                 'The order of inputs in converted model will match the order of '
                 'specified inputs. The shape and value are '
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
        'extension':
            {'description':
                 'Paths or a comma-separated list of paths to libraries '
                 '(.so or .dll) with extensions. For the legacy MO path '
                 '(if `--use_legacy_frontend` is used), a directory or a '
                 'comma-separated list of directories with extensions '
                 'are supported. To disable all extensions including '
                 'those that are placed at the default location, pass an empty string.',
             'action': CanonicalizePathCheckExistenceAction,
             'type': readable_files_or_empty},
        'version':
            {'action': 'version',
            #FIXME: Why the following is not accessible from arg parser?
             'version': 'OpenVINO Model Converter (ovc) {}'.format(VersionChecker().get_ie_version())},
    }
