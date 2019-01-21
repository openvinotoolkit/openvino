"""
 Copyright (c) 2018 Intel Corporation

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

HELP_MESSAGES = {
    'IMAGE_MESSAGE': "Path to a folder with images or to image files.",
    'MULTI_INPUT_MESSAGE': "Path to multi input file containing.",
    'MODEL_MESSAGE': "Path to an .xml file with a trained model.",
    'PLUGIN_PATH_MESSAGE': "Path to a plugin folder.",
    'API_MESSAGE': "Enable using sync/async API. Default value is sync",
    'TARGET_DEVICE_MESSAGE': "Specify a target device to infer on: CPU, GPU, FPGA or MYRIAD. "
                           "Use \"-d HETERO:<comma separated devices list>\" format to specify HETERO plugin. "
    "The application looks for a suitable plugin for the specified device.",
    'ITERATIONS_COUNT_MESSAGE': "Number of iterations. "
    "If not specified, the number of iterations is calculated depending on a device.",
    'INFER_REQUESTS_COUNT_MESSAGE': "Number of infer requests (default value is 2).",
    'INFER_NUM_THREADS_MESSAGE': "Number of threads to use for inference on the CPU "
                                 "(including Hetero cases).",
    'CUSTOM_CPU_LIBRARY_MESSAGE': "Required for CPU custom layers. "
                                  "Absolute path to a shared library with the kernels implementations.",
    'CUSTOM_GPU_LIBRARY_MESSAGE': "Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.",
    'BATCH_SIZE_MESSAGE': "Optional. Batch size value. If not specified, the batch size value is determined from IR",
    'INFER_THREADS_PINNING_MESSAGE': "Optional. Enable (\"YES\" is default value) or disable (\"NO\")"
                                     "CPU threads pinning for CPU-involved inference."
}

DEVICE_DURATION_IN_SECS = {
    "CPU": 60,
    "GPU": 60,
    "VPU": 60,
    "MYRIAD": 60,
    "FPGA": 120,
    "HDDL": 60,
    "UNKNOWN": 120
}

IMAGE_EXTENSIONS = ['JPEG', 'JPG', 'PNG', 'BMP']

MYRIAD_DEVICE_NAME = "MYRIAD"
CPU_DEVICE_NAME = "CPU"
GPU_DEVICE_NAME = "GPU"
UNKNOWN_DEVICE_TYPE = "UNKNOWN"

BATCH_SIZE_ELEM = 0

LAYOUT_TYPE = 'NCHW'

XML_EXTENSION = ".xml"
BIN_EXTENSION = ".bin"

XML_EXTENSION_PATTERN = '*' + XML_EXTENSION
