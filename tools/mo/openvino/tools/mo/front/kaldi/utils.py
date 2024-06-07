# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import logging as log
import os

import numpy as np

from openvino.tools.mo.front.kaldi.loader.utils import read_placeholder, read_binary_integer32_token, read_blob, read_token_value, \
    find_next_tag
from openvino.tools.mo.utils.error import Error


def read_binary_matrix(file_desc: io.BufferedReader, read_token: bool = True):
    if read_token:
        read_placeholder(file_desc)
    rows_number = read_binary_integer32_token(file_desc)
    cols_number = read_binary_integer32_token(file_desc)
    # to compare: ((float *)a->buffer())[10]
    return read_blob(file_desc, rows_number * cols_number), (rows_number, cols_number)


def read_binary_vector(file_desc: io.BufferedReader, read_token: bool = True, dtype=np.float32):
    if read_token:
        read_placeholder(file_desc)
    elements_number = read_binary_integer32_token(file_desc)
    return read_blob(file_desc, elements_number, dtype)


def read_binary_vector_of_pairs(file_desc: io.BufferedReader, read_token: bool = True, dtype=np.float32):
    if read_token:
        read_placeholder(file_desc)
    elements_number = read_binary_integer32_token(file_desc)
    return read_blob(file_desc, 2 * elements_number, dtype)


def read_learning_info(pb: io.BufferedReader):
    while True:
        read_placeholder(pb, 1)
        first_char = pb.read(1)
        pb.seek(-2, os.SEEK_CUR)
        position = pb.tell()
        if first_char == b'L':
            cur_pos = pb.tell()
            token = find_next_tag(pb)
            pb.seek(cur_pos)
            if token in ['<LearnRateCoef>', '<LearningRate>']:
                token = bytes(token, 'ascii')
            else:
                log.debug('Unexpected tag: {}'.format(token))
                break
        elif first_char == b'B':
            token = b'<BiasLearnRateCoef>'
        elif first_char == b'M':
            token = b'<MaxNorm>'
        elif first_char == b'!':  # token = b'<EndOfComponent>'
            break
        else:
            break
        try:
            read_token_value(pb, token)
        except Error:
            pb.seek(position)
            break

