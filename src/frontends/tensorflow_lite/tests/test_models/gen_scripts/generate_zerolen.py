# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

path_to_model_dir = os.path.join(sys.argv[1], "zerolen")
tflite_file_name = 'zerolen.tflite'
tflite_model_path = os.path.join(path_to_model_dir, tflite_file_name)
if not os.path.exists(path_to_model_dir):
    os.mkdir(path_to_model_dir)
with open(tflite_model_path, 'wb') as f:
    f.close()
