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
import io
import numpy as np
import os
import logging as log

from mo.front.kaldi.loader.utils import read_placeholder, read_binary_integer32_token, read_blob, read_token_value, find_next_tag
from mo.utils.error import Error


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

