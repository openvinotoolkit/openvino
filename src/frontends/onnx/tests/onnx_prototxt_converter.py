#!/usr/bin/env python

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
from onnx.external_data_helper import convert_model_to_external_data
import os
import sys

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


def save_model(proto, f, format=None, save_as_external_data=False, all_tensors_to_one_file=True, location=None, size_threshold=1024, convert_attribute=False):
    if isinstance(proto, bytes):
        proto = onnx.serialization.registry.get("protobuf").serialize_proto(proto, onnx.ModelProto())

    if save_as_external_data:
        convert_model_to_external_data(proto, all_tensors_to_one_file, location, size_threshold, convert_attribute)

    s = onnx.serialization.registry.get("protobuf").serialize_proto(proto)
    onnx._save_bytes(s, f)


if __name__ == '__main__':
    args = docopt(__doc__)
    input_file_path = args['INPUT_FILE']
    if not args['OUTPUT_FILE']:
        output_file_path = _get_output_file_path(*os.path.splitext(input_file_path))
    else:
        output_file_path = args['OUTPUT_FILE']

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
        save_model(converted_model, output_file_path)
    else:
        sys.exit('ERROR: Provided input or output file has unsupported format.')
