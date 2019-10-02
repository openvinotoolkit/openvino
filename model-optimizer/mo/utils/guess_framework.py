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

import re
from argparse import Namespace

from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


def deduce_framework_by_namespace(argv: Namespace):
    if not argv.framework:
        if getattr(argv, 'saved_model_dir', None) or getattr(argv, 'input_meta_graph', None):
            argv.framework = 'tf'
        elif getattr(argv, 'input_symbol', None) or getattr(argv, 'pretrained_model_name', None):
            argv.framework = 'mxnet'
        elif getattr(argv, 'input_proto', None):
            argv.framework = 'caffe'
        elif argv.input_model is None:
            raise Error('Path to input model is required: use --input_model.')
        else:
            argv.framework = guess_framework_by_ext(argv.input_model)
        if not argv.framework:
            raise Error('Framework name can not be deduced from the given options: {}={}. Use --framework to choose '
                        'one of caffe, tf, mxnet, kaldi, onnx', '--input_model', argv.input_model, refer_to_faq_msg(15))

    return map(lambda x: argv.framework == x, ['tf', 'caffe', 'mxnet', 'kaldi', 'onnx'])


def guess_framework_by_ext(input_model_path: str) -> int:
    if re.match('^.*\.caffemodel$', input_model_path):
        return 'caffe'
    elif re.match('^.*\.pb$', input_model_path):
        return 'tf'
    elif re.match('^.*\.pbtxt$', input_model_path):
        return 'tf'
    elif re.match('^.*\.params$', input_model_path):
        return 'mxnet'
    elif re.match('^.*\.nnet$', input_model_path):
        return 'kaldi'
    elif re.match('^.*\.mdl', input_model_path):
        return 'kaldi'
    elif re.match('^.*\.onnx$', input_model_path):
        return 'onnx'
