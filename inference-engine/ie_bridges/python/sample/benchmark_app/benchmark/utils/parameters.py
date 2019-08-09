"""
 Copyright (C) 2018-2019 Intel Corporation

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

import argparse
from fnmatch import fnmatch

XML_EXTENSION = ".xml"
BIN_EXTENSION = ".bin"

XML_EXTENSION_PATTERN = '*' + XML_EXTENSION

def validate_args(args):
    if args.number_iterations is not None and args.number_iterations < 0:
        raise Exception("Number of iterations should be positive (invalid -niter option value)")
    if args.number_infer_requests and args.number_infer_requests < 0:
        raise Exception("Number of inference requests should be positive (invalid -nireq option value)")
    if not fnmatch(args.path_to_model, XML_EXTENSION_PATTERN):
        raise Exception('Path {} is not xml file.')

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help="Show this help message and exit.")
    args.add_argument('-i', '--path_to_input', type=str, required=False,
                      help="Optional. Path to a folder with images and/or binaries or to specific image or binary file.")
    args.add_argument('-m', '--path_to_model', type=str, required=True,
                      help="Required. Path to an .xml file with a trained model.")
    args.add_argument('-d', '--target_device', type=str, required=False, default="CPU",
                      help="Optional. Specify a target device to infer on: CPU, GPU, FPGA, HDDL or MYRIAD. "
                           "Use \"-d HETERO:<comma separated devices list>\" format to specify HETERO plugin. ")
    args.add_argument('-l', '--path_to_extension', type=str, required=False, default=None,
                      help="Optional. Required for CPU custom layers. "
                           "Absolute path to a shared library with the kernels implementations.")
    args.add_argument('-c', '--path_to_cldnn_config', type=str, required=False,
                      help="Optional. Required for GPU custom kernels. Absolute path to an .xml file with the "
                           "kernels description.")
    args.add_argument('-api', '--api_type', type=str, required=False, default='async', choices=['sync', 'async'],
                      help="Optional. Enable using sync/async API. Default value is async.")
    args.add_argument('-niter', '--number_iterations', type=int, required=False, default=None,
                      help="Optional. Number of iterations. "
                           "If not specified, the number of iterations is calculated depending on a device.")
    args.add_argument('-nireq', '--number_infer_requests', type=int, required=False, default=None,
                      help="Optional. Number of infer requests. Default value is determined automatically for device.")
    args.add_argument('-b', '--batch_size', type=int, required=False, default=None,
                      help="Optional. Batch size value. If not specified, the batch size value is determined from Intermediate Representation")
    args.add_argument('-stream_output', type=str2bool, required=False, default=False, nargs='?', const=True,
                      help="Optional. Print progress as a plain text. When specified, an interactive progress bar is replaced with a "
                           "multiline output.")
    args.add_argument('-t', '--time', type=int, required=False, default=None,
                      help="Optional. Time in seconds to execute topology.")
    args.add_argument('-progress', type=str2bool, required=False, default=False, nargs='?', const=True,
                      help="Optional. Show progress bar (can affect performance measurement). Default values is \"False\".")
    args.add_argument('-nstreams', '--number_streams', type=str, required=False, default=None,
                      help="Optional. Number of streams to use for inference on the CPU/GPU in throughput mode "
                           "(for HETERO device case use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).")
    args.add_argument('-nthreads', '--number_threads', type=int, required=False, default=None,
                      help="Number of threads to use for inference on the CPU "
                           "(including HETERO case).")
    args.add_argument('-pin', '--infer_threads_pinning', type=str, required=False, default='YES', choices=['YES', 'NO'],
                      help="Optional. Enable (\"YES\" is default value) or disable (\"NO\")"
                      "CPU threads pinning for CPU-involved inference.")
    args.add_argument('--exec_graph_path', type=str, required=False,
                      help="Optional. Path to a file where to store executable graph information serialized.")
    args.add_argument("-pc", "--perf_counts", type=str2bool, required=False, default=False, nargs='?', const=True,
                      help="Optional. Report performance counters.", )
    parsed_args = parser.parse_args()

    validate_args(parsed_args)

    return parsed_args
