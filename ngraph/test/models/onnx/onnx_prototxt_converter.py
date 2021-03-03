#!/usr/bin/env python
# *****************************************************************************
#  Copyright 2017-2021 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# *****************************************************************************
"""Converts protobuf files from binary message format into prototxt format and vice-versa.

Supports files with only '.onnx' or '.prototxt' extensions. Application may accept only single
argument denoting input file. In that case it converts it to the second message format based on the
extension of argument.

Usage:
  onnx_prototxt_converter.py INPUT_FILE [OUTPUT_FILE]

Arguments:
  INPUT_FILE   The path for the input model file.
  OUTPUT_FILE  The path for the converted model file.

Options:
  -h --help            show this help message and exit
"""


from docopt import docopt
from google.protobuf import text_format
import onnx
import os

ONNX_SUFFX = '.onnx'
PROTOTXT_SUFFX = '.prototxt'

def _bin2txt(model):
    return text_format.MessageToString(model, as_utf8=True, float_format='.17g')

def _txt2bin(model):
    m_proto = onnx.ModelProto()
    text_format.Parse(model, m_proto, allow_field_number=True)
    return m_proto

def _is_bin_file(path):
    # check file extension
    return os.path.splitext(path)[1] == ONNX_SUFFX

def _is_txt_file(path):
    # check file extension
    return os.path.splitext(path)[1] == PROTOTXT_SUFFX

_ext_map = {
    '.onnx': '.prototxt',
    '.prototxt': '.onnx',
}

def _get_output_file_path(path, extension):
    return path + _ext_map[extension]

if __name__ == '__main__':
    args = docopt(__doc__)
    input_file_path = args['INPUT_FILE']
    if not args['OUTPUT_FILE']:
        output_file_path = _get_output_file_path(*os.path.splitext(input_file_path))
    else:
        output_file_path = args['OUTPUT_FILE']

    print('Converting {} to {}.'.format(input_file_path, output_file_path))

    if not os.path.exists(input_file_path):
        sys.exit('ERROR: Provided input model path does not exists: {}'.format(input_file_path))

    # convert from binary format to text format
    if _is_bin_file(input_file_path) and _is_txt_file(output_file_path):
        str_msg = _bin2txt(onnx.load_model(input_file_path))
        with open(output_file_path, 'w') as f:
            f.write(str_msg)
    # convert from text format to binary format
    elif _is_txt_file(input_file_path) and _is_bin_file(output_file_path):
        with open(input_file_path, 'r') as f:
            converted_model = _txt2bin(f.read())
        onnx.save(converted_model, output_file_path)
    else:
        sys.exit('ERROR: Provided input or output file has unsupported format.')
