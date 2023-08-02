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
        'input':
            {'description':
                 'Information of model input required for model conversion. '
                 'This is a comma separated list with optional '
                 'input names, shapes and data types. The order of inputs '
                 'in converted model will match the order of '
                 'specified inputs. The shape is specified as comma-separated list. '
                 'The data type of input node is specified in braces and can have one of '
                 'the values: f64, f32, f16, i64, i32, u8, boolean. If data type is not '
                 'specified explicitly then there are two options: if '
                 'input node is a parameter, data type is taken from the '
                 'original node data type, if input node is not a parameter, '
                 'data type is set to f32. Example, to set `input_1` input '
                 'with shape [1,100] and float32 type, and `sequence_len` input '
                 'with int32 type \"input_1[1,100]{f32},sequence_len{i32}\".'},
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
