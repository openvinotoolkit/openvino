# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def get_convert_model_help_specifics():
    from openvino.tools.ovc.cli_parser import CanonicalizePathCheckExistenceAction, readable_dirs_or_files_or_empty, readable_files_or_empty
    from openvino.tools.ovc.version import VersionChecker
    return {
        'input_model':
            {'description':
                 'Input model file(s) from TensorFlow, ONNX, PaddlePaddle. '
                 'Use openvino.convert_model in Python to convert models from PyTorch.'
                 '',
             'action': CanonicalizePathCheckExistenceAction,
             'type': readable_dirs_or_files_or_empty,
             'aliases': {}},
        'input':
            {'description':
                 'Information of model input required for model conversion. '
                 'This is a comma separated list with optional '
                 'input names and shapes. The order of inputs '
                 'in converted model will match the order of '
                 'specified inputs. The shape is specified as comma-separated list. '
                 'Example, to set `input_1` input with shape [1,100] and `sequence_len` input '
                 'with shape [1,?]: \"input_1[1,100],sequence_len[1,?]\", where "?" is a dynamic dimension, '
                 'which means that such a dimension can be specified later in the runtime. '
                 'If the dimension is set as an integer (like 100 in [1,100]), such a dimension is not supposed '
                 'to be changed later, during a model conversion it is treated as a static value. '
                 'Example with unnamed inputs: \"[1,100],[1,?]\".'},
        'output':
            {'description':
                'The name of the output operation of the model or list of names. For TensorFlow*, '
                'do not add :0 to this name. The order of outputs in converted model is the '
                'same as order of specified operation names. Outputs should be separated with comma (spaces are ignored): '
                'Example 1: ovc ... output="out_1, out_2". '
                'Example 2: ovc ... output="x,y,z" equivalent to ovc ... output="x, y, z".'},
        'extension':
            {'description':
                 'Paths or a comma-separated list of paths to libraries '
                 '(.so or .dll) with extensions. To disable all extensions including '
                 'those that are placed at the default location, pass an empty string.',
             'action': CanonicalizePathCheckExistenceAction,
             'type': readable_files_or_empty},
        'version':
            {'action': 'version',
            #FIXME: Why the following is not accessible from arg parser?
             'version': 'OpenVINO Model Converter (ovc) {}'.format(VersionChecker().get_ie_version())},
    }
