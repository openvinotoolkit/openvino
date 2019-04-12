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
