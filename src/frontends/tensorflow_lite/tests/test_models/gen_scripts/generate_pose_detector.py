# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This is a way to test Densify operation. Should be removed after enabling Layer-based test

import numpy as np
import os
import sys
import urllib.request
import zipfile

path_to_model_dir = os.path.join(sys.argv[1], "downloads")
tflite_file_name = 'pose_detector.tflite'
tflite_model_path = os.path.join(path_to_model_dir, tflite_file_name)
task_file_name = 'pose_landmarker_lite.task'
task_file_path = os.path.join(path_to_model_dir, task_file_name)
if not os.path.exists(path_to_model_dir):
    os.makedirs(path_to_model_dir)
if not os.path.exists(tflite_model_path):
    if not os.path.exists(task_file_path):
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task", task_file_path)
    with zipfile.ZipFile(task_file_path, "r") as f:
        f.extract(tflite_file_name, path_to_model_dir)

