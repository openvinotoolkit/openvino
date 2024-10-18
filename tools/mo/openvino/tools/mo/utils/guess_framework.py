# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
from argparse import Namespace

from openvino.tools.mo.utils.error import Error


def deduce_legacy_frontend_by_namespace(argv: Namespace):
    if not hasattr(argv, 'framework') or not argv.framework:
        if getattr(argv, 'saved_model_dir', None) or getattr(argv, 'input_meta_graph', None):
            argv.framework = 'tf'
        elif getattr(argv, 'input_proto', None):
            argv.framework = 'caffe'
        elif argv.input_model is None:
            raise Error('Path to input model is required: use --input_model.')
        else:
            argv.framework = guess_framework_by_ext(argv.input_model)

    return map(lambda x: argv.framework == x, ['tf', 'caffe', 'kaldi', 'onnx'])


def guess_framework_by_ext(input_model_path: str) -> int:
    if re.match(r'^.*\.caffemodel$', input_model_path):
        return 'caffe'
    elif re.match(r'^.*\.pb$', input_model_path):
        return 'tf'
    elif re.match(r'^.*\.pbtxt$', input_model_path):
        return 'tf'
    elif re.match(r'^.*\.nnet$', input_model_path):
        return 'kaldi'
    elif re.match(r'^.*\.mdl', input_model_path):
        return 'kaldi'
    elif re.match(r'^.*\.onnx$', input_model_path):
        return 'onnx'
