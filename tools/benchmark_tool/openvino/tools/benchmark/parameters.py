# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys,argparse
from fnmatch import fnmatch

from openvino.tools.benchmark.utils.utils import show_available_devices

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue

class print_help(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()
        show_available_devices()
        sys.exit()

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action=print_help, nargs='?', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-i', '--paths_to_input', action='append', nargs='+', type=str, required=False,
                      help='Optional. '
                           'Path to a folder with images and/or binaries or to specific image or binary file.'
                           'It is also allowed to map files to network inputs: '
                           'input_1:file_1/dir1,file_2/dir2,input_4:file_4/dir4 input_2:file_3/dir3')
    args.add_argument('-m', '--path_to_model', type=str, required=True,
                      help='Required. Path to an .xml/.onnx file with a trained model or '
                           'to a .blob file with a trained compiled model.')
    args.add_argument('-d', '--target_device', type=str, required=False, default='CPU',
                      help='Optional. Specify a target device to infer on (the list of available devices is shown below). '
                           'Default value is CPU. Use \'-d HETERO:<comma separated devices list>\' format to specify HETERO plugin. '
                           'Use \'-d MULTI:<comma separated devices list>\' format to specify MULTI plugin. '
                           'The application looks for a suitable plugin for the specified device.')
    args.add_argument('-l', '--path_to_extension', type=str, required=False, default=None,
                      help='Optional. Required for CPU custom layers. '
                           'Absolute path to a shared library with the kernels implementations.')
    args.add_argument('-c', '--path_to_cldnn_config', type=str, required=False,
                      help='Optional. Required for GPU custom kernels. Absolute path to an .xml file with the '
                           'kernels description.')
    args.add_argument('-hint', '--perf_hint', type=str, required=False, default='', choices=['throughput', 'cumulative_throughput', 'latency', 'none'],
                      help='Optional. Performance hint (latency or throughput or cumulative_throughput or none). Performance hint allows the OpenVINO device to select the right network-specific settings.\n'
                            '\'throughput\': device performance mode will be set to THROUGHPUT. \n'
                            '\'cumulative_throughput\': device performance mode will be set to CUMULATIVE_THROUGHPUT. \n'
                            '\'latency\': device performance mode will be set to LATENCY. \n'
                            '\'none\': no device performance mode will be set. \n'
                            'Using explicit \'nstreams\' or other device-specific options, please set hint to \'none\'')
    args.add_argument('-api', '--api_type', type=str, required=False, default='async', choices=['sync', 'async'],
                      help='Optional. Enable using sync/async API. Default value is async.')
    args.add_argument('-niter', '--number_iterations', type=check_positive, required=False, default=None,
                      help='Optional. Number of iterations. '
                           'If not specified, the number of iterations is calculated depending on a device.')
    args.add_argument('-nireq', '--number_infer_requests', type=check_positive, required=False, default=0,
                      help='Optional. Number of infer requests. Default value is determined automatically for device.')
    args.add_argument('-b', '--batch_size', type=str, required=False, default='',
                      help='Optional. ' +
                           'Batch size value. ' +
                           'If not specified, the batch size value is determined from Intermediate Representation')
    args.add_argument('-stream_output', type=str2bool, required=False, default=False, nargs='?', const=True,
                      help='Optional. '
                           'Print progress as a plain text. '
                           'When specified, an interactive progress bar is replaced with a multi-line output.')
    args.add_argument('-t', '--time', type=int, required=False, default=None,
                      help='Optional. Time in seconds to execute topology.')
    args.add_argument('-progress', type=str2bool, required=False, default=False, nargs='?', const=True,
                      help='Optional. '
                           'Show progress bar (can affect performance measurement). Default values is \'False\'.')
    args.add_argument('-shape', type=str, required=False, default='',
                      help='Optional. '
                           'Set shape for input. For example, "input1[1,3,224,224],input2[1,4]" or "[1,3,224,224]" in case of one input size.'
                           'This parameter affect model Parameter shape, can be dynamic. For dynamic dimesions use symbol `?`, `-1` or range `low.. up`.')
    args.add_argument('-data_shape', type=str, required=False, default='',
                      help='Optional. '
                           'Optional if network shapes are all static (original ones or set by -shape).'
                           'Required if at least one input shape is dynamic and input images are not provided.'
                           'Set shape for input tensors. For example, "input1[1,3,224,224][1,3,448,448],input2[1,4][1,8]" or "[1,3,224,224][1,3,448,448] in case of one input size.')
    args.add_argument('-layout', type=str, required=False, default='',
                      help='Optional. '
                           'Prompts how network layouts should be treated by application. '
                           'For example, "input1[NCHW],input2[NC]" or "[NCHW]" in case of one input size.')
    args.add_argument('-nstreams', '--number_streams', type=str, required=False, default=None,
                      help='Optional. Number of streams to use for inference on the CPU/GPU/MYRIAD '
                           '(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> '
                           'or just <nstreams>). '
                           'Default value is determined automatically for a device. Please note that although the automatic selection '
                           'usually provides a reasonable performance, it still may be non - optimal for some cases, especially for very small networks. '
                           'Also, using nstreams>1 is inherently throughput-oriented option, while for the best-latency '
                           'estimations the number of streams should be set to 1. '
                           'See samples README for more details.')
    args.add_argument('--latency_percentile', type=int, required=False, default=50, choices=range(1,101),
                      help='Optional. Defines the percentile to be reported in latency metric. The valid range is [1, 100]. The default value is 50 (median).')
    args.add_argument('-nthreads', '--number_threads', type=int, required=False, default=None,
                      help='Number of threads to use for inference on the CPU, GNA '
                           '(including HETERO and MULTI cases).')
    args.add_argument('-pin', '--infer_threads_pinning', type=str, required=False,  choices=['YES', 'NO', 'NUMA', 'HYBRID_AWARE'],
                      help='Optional. Enable  threads->cores (\'YES\' which is OpenVINO runtime\'s default for conventional CPUs), '
                           'threads->(NUMA)nodes (\'NUMA\'), '
                           'threads->appropriate core types (\'HYBRID_AWARE\', which is OpenVINO runtime\'s default for Hybrid CPUs) '
                           'or completely disable (\'NO\') '
                           'CPU threads pinning for CPU-involved inference.')
    args.add_argument('-exec_graph_path', '--exec_graph_path', type=str, required=False,
                      help='Optional. Path to a file where to store executable graph information serialized.')
    args.add_argument('-pc', '--perf_counts', type=str2bool, required=False, default=False, nargs='?', const=True,
                      help='Optional. Report performance counters.', )
    args.add_argument('-pcseq', '--pcseq', type=str2bool, required=False, default=False, nargs='?', const=True,
                      help='Optional. Report latencies for each shape in -data_shape sequence.', )
    args.add_argument('-inference_only', '--inference_only', type=str2bool, required=False, default=None, nargs='?', const=True,
                      help='Optional. If true inputs filling only once before measurements (default for static models), '
                                     'else inputs filling is included into loop measurement (default for dynamic models)', )
    args.add_argument('-report_type', '--report_type', type=str, required=False,
                      choices=['no_counters', 'average_counters', 'detailed_counters'],
                      help="Optional. Enable collecting statistics report. \"no_counters\" report contains "
                           "configuration options specified, resulting FPS and latency. \"average_counters\" "
                           "report extends \"no_counters\" report and additionally includes average PM "
                           "counters values for each layer from the network. \"detailed_counters\" report "
                           "extends \"average_counters\" report and additionally includes per-layer PM "
                           "counters and latency for each executed infer request.")
    args.add_argument('-report_folder', '--report_folder', type=str, required=False, default='',
                      help="Optional. Path to a folder where statistics report is stored.")
    args.add_argument('-dump_config', type=str, required=False, default='',
                      help="Optional. Path to JSON file to dump OpenVINO parameters, which were set by application.")
    args.add_argument('-load_config', type=str, required=False, default='',
                      help="Optional. Path to JSON file to load custom OpenVINO parameters."
                           " Please note, command line parameters have higher priority then parameters from configuration file.")
    args.add_argument('-infer_precision', type=str, required=False,
                      help='Optional. Hint to specifies inference precision. Example: -infer_precision CPU:bf16,GPU:f32')
    args.add_argument('-ip', '--input_precision', type=str, required=False, choices=['u8', 'U8', 'f16','FP16', 'f32','FP32'],
                      help='Optional. Specifies precision for all input layers of the network.')
    args.add_argument('-op', '--output_precision', type=str, required=False, choices=['u8', 'U8', 'f16','FP16', 'f32','FP32'],
                      help='Optional. Specifies precision for all output layers of the network.')
    args.add_argument('-iop', '--input_output_precision', type=str, required=False,
                      help='Optional. Specifies precision for input and output layers by name. Example: -iop "input:f16, output:f16". Notice that quotes are required. Overwrites precision from ip and op options for specified layers.')
    args.add_argument('-cdir', '--cache_dir', type=str, required=False, default='',
                      help="Optional. Enable model caching to specified directory")
    args.add_argument('-lfile', '--load_from_file', required=False, nargs='?', default=argparse.SUPPRESS,
                      help="Optional. Loads model from file directly without read_network.")
    args.add_argument('-iscale', '--input_scale', type=str, required=False, default='',
                      help="Optional. Scale values to be used for the input image per channel.\n Values to be provided in the [R, G, B] format. Can be defined for desired input of the model.\n"
                           "Example: -iscale data[255,255,255],info[255,255,255]\n")
    args.add_argument('-imean', '--input_mean', type=str, required=False, default='',
                      help="Optional. Mean values to be used for the input image per channel.\n Values to be provided in the [R, G, B] format. Can be defined for desired input of the model.\n"
                           "Example: -imean data[255,255,255],info[255,255,255]\n")
    args.add_argument('-expconv', '--experiment_convolution', type=str2bool, required=False, default=False, nargs='?', const=True,
                      help="Optional. Enable new experiment convolution algorithm. Only valid in CPU plugin.")
    parsed_args = parser.parse_args()

    return parsed_args
