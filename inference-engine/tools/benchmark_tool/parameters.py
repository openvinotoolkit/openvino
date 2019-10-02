import argparse
from fnmatch import fnmatch

from openvino.tools.benchmark.utils.constants import XML_EXTENSION_PATTERN


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def validate_args(args):
    if args.number_iterations is not None and args.number_iterations < 0:
        raise Exception("Number of iterations should be positive (invalid -niter option value)")
    if args.number_infer_requests and args.number_infer_requests < 0:
        raise Exception("Number of inference requests should be positive (invalid -nireq option value)")
    if not fnmatch(args.path_to_model, XML_EXTENSION_PATTERN):
        raise Exception('Path {} is not xml file.')


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-i', '--path_to_input', type=str, required=False,
                      help='Optional. '
                           'Path to a folder with images and/or binaries or to specific image or binary file.')
    args.add_argument('-m', '--path_to_model', type=str, required=True,
                      help='Required. Path to an .xml file with a trained model.')
    args.add_argument('-d', '--target_device', type=str, required=False, default='CPU',
                      help='Optional. Specify a target device to infer on: CPU, GPU, FPGA, HDDL or MYRIAD. '
                           'Use \'-d HETERO:<comma separated devices list>\' format to specify HETERO plugin. '
                           'Use \'-d MULTI:<comma separated devices list>\' format to specify MULTI plugin. '
                           'The application looks for a suitable plugin for the specified device.')
    args.add_argument('-l', '--path_to_extension', type=str, required=False, default=None,
                      help='Optional. Required for CPU custom layers. '
                           'Absolute path to a shared library with the kernels implementations.')
    args.add_argument('-c', '--path_to_cldnn_config', type=str, required=False,
                      help='Optional. Required for GPU custom kernels. Absolute path to an .xml file with the '
                           'kernels description.')
    args.add_argument('-api', '--api_type', type=str, required=False, default='async', choices=['sync', 'async'],
                      help='Optional. Enable using sync/async API. Default value is async.')
    args.add_argument('-niter', '--number_iterations', type=int, required=False, default=None,
                      help='Optional. Number of iterations. '
                           'If not specified, the number of iterations is calculated depending on a device.')
    args.add_argument('-nireq', '--number_infer_requests', type=int, required=False, default=None,
                      help='Optional. Number of infer requests. Default value is determined automatically for device.')
    args.add_argument('-b', '--batch_size', type=int, required=False, default=None,
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
    args.add_argument('-nstreams', '--number_streams', type=str, required=False, default=None,
                      help='Optional. Number of streams to use for inference on the CPU/GPU in throughput mode '
                           '(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> '
                           'or just <nstreams>). '
                           'Default value is determined automatically for a device. Please note that although the automatic selection '
                           'usually provides a reasonable performance, it still may be non - optimal for some cases, especially for very small networks. '
                           'See samples README for more details.')

    args.add_argument('-nthreads', '--number_threads', type=int, required=False, default=None,
                      help='Number of threads to use for inference on the CPU '
                           '(including HETERO and MULTI cases).')
    args.add_argument('-pin', '--infer_threads_pinning', type=str, required=False, default='YES', choices=['YES', 'NO'],
                      help='Optional. Enable (\'YES\' is default value) or disable (\'NO\')'
                           'CPU threads pinning for CPU-involved inference.')
    args.add_argument('--exec_graph_path', type=str, required=False,
                      help='Optional. Path to a file where to store executable graph information serialized.')
    args.add_argument('-pc', '--perf_counts', type=str2bool, required=False, default=False, nargs='?', const=True,
                      help='Optional. Report performance counters.', )
    args.add_argument('--report_type', type=str, required=False,
                      choices=['no_counters', 'average_counters', 'detailed_counters'],
                      help="Optional. Enable collecting statistics report. \"no_counters\" report contains "
                           "configuration options specified, resulting FPS and latency. \"average_counters\" "
                           "report extends \"no_counters\" report and additionally includes average PM "
                           "counters values for each layer from the network. \"detailed_counters\" report "
                           "extends \"average_counters\" report and additionally includes per-layer PM "
                           "counters and latency for each executed infer request.")
    args.add_argument('--report_folder', type=str, required=False, default='',
                      help="Optional. Path to a folder where statistics report is stored.")
    parsed_args = parser.parse_args()

    validate_args(parsed_args)

    return parsed_args
