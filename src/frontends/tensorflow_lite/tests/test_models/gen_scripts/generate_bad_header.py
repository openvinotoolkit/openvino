# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

path_to_model_dir = os.path.join(sys.argv[1], "bad_header")
if not os.path.exists(path_to_model_dir):
    os.makedirs(path_to_model_dir, exist_ok=True)

# Correct FOURCC is 'TFL3', it should be in first 4 bytes or
# in second 4 bytes in case of size prefixed

# 0. File has zero length
tflite_file_name = 'zerolen.tflite'
tflite_model_path = os.path.join(path_to_model_dir, tflite_file_name)
if not os.path.exists(path_to_model_dir):
    os.mkdir(path_to_model_dir)
with open(tflite_model_path, 'wb') as f:
    f.close()

# 1. File length is less than 4 bytes
tflite_file_name = 'wrong_len_3.tflite'
tflite_model_path = os.path.join(path_to_model_dir, tflite_file_name)
with open(tflite_model_path, 'wb') as f:
    f.write(bytearray(b'TFL'))
    f.close()

# 2. File length is enough, but FOURCC isn't aligned as expected
tflite_file_name = 'wrong_pos.tflite'
tflite_model_path = os.path.join(path_to_model_dir, tflite_file_name)
with open(tflite_model_path, 'wb') as f:
    f.write(bytearray(b'   TFL3 '))
    f.close()
