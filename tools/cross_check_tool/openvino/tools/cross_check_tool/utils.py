# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log
import os
import sys
import traceback
from pathlib import Path

from openvino.runtime import Layout, ProfilingInfo, Output
from openvino.runtime.utils.types import get_dtype

try:
    import cv2
except Exception as e:
    log.error(f"Can not import OpenCV Python package.\nPlease install required python packages by running:\n"
              f"pip3 install -r requirements.txt\n\n Original error message: {e}")
    sys.exit(1)

try:
    import numpy as np
except Exception as e:
    log.error(f"Can not import numpy python package.\nPlease install required python packages by running:\n"
              f"pip3 install -r requirements.txt\n\n Original error message: {e}")
    sys.exit(1)

verbosity = False


def set_verbosity(flag: bool):
    global verbosity
    verbosity = flag


###
#   USER INTERACTION
###


class LvlFormatter(log.Formatter):
    usual = '[ %(levelname)s ]  %(msg)s'
    format_dict = {
        'no_lvl': '%(msg)s',
        log.DEBUG: '[ %(asctime)s ] [ %(levelname)s ] [ %(module)s:%(lineno)d ]  %(msg)s',
        log.INFO: usual, log.WARNING: usual, log.ERROR: usual, log.CRITICAL: usual
    }

    def __init__(self, lvl, fmt=None):
        log.Formatter.__init__(self, fmt)
        self.lvl = lvl

    def format(self, record: log.LogRecord):
        if self.lvl == 'DEBUG':
            self._style._fmt = self.format_dict[log.DEBUG]
        else:
            self._style._fmt = self.format_dict[record.levelno]
        if 'no_lvl' in record.__dict__.keys() and record.__dict__['no_lvl']:
            self._style._fmt = self.format_dict['no_lvl']
        return log.Formatter.format(self, record)


def set_logger(lvl: str):
    logger = log.getLogger()
    logger.setLevel(lvl)
    handler = log.StreamHandler(sys.stdout)
    handler.setFormatter(LvlFormatter(lvl))
    logger.addHandler(handler)


def error_handling(desc: str):
    """
    Error handler that prints description formatted with keyword arguments in case of exception
    :param desc: description for an error
    :return: decorator
    """

    def decorator(func):
        def try_except_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                exception_type = type(e).__name__
                log.error(f"The following error happened while {desc.format(**kwargs)}:\n[ {exception_type} ] {e}")
                global verbosity
                if verbosity:
                    traceback.print_tb(tb=e.__traceback__, file=sys.stdout)
                sys.exit(1)

        return try_except_func

    return decorator


class ExistingFileAction(argparse.Action):
    """
    Expand user home directory paths and convert relative-paths to absolute.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            if not os.path.isfile(values):
                log.error(f"File was not found: {values}")
                sys.exit(1)
        setattr(namespace, self.dest, values)


class ExistingDirAction(argparse.Action):
    """
    Expand user home directory paths and convert relative-paths to absolute.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            if not os.path.isdir(values):
                log.error(f"Directory was not found: {values}")
                sys.exit(1)
        setattr(namespace, self.dest, values)


def build_parser():
    parser = argparse.ArgumentParser(
        prog='Cross Check Tool',
        description='Cross Check Tool is a console application that enables comparing accuracy and provides performance'
                    ' metrics',
        usage='\n' + '-' * 62 +
              '\nFor cross precision check provide two IRs \n'
              '(mapping files may be needed) run:'
              '\npython3 cross_check_tool.py                        \\'
              '\n--input path/to/file/describing/input              \\'
              '\n--model path/to/model/*.xml                        \\'
              '\n--device device_for_model                          \\'
              '\n--reference_model path/to/reference_model/*.xml    \\'
              '\n--reference_device reference_device_for_model      \n'
              + '-' * 62 +
              '\nFor cross device check with one precision provide one IR run:'
              '\npython3 cross_check_tool.py                        \\'
              '\n--input path/to/file/describing/input              \\'
              '\n--model path/to/model/*.xml                        \\'
              '\n--device device_for_model                          \\'
              '\n--reference_device reference_device_for_model      \n'
              + '-' * 62 +
              '\nFor dumping tensors and performance counters run:'
              '\npython3 cross_check_tool.py                        \\'
              '\n--input path/to/file/describing/input              \\'
              '\n--model path/to/model/*.xml                        \\'
              '\n--device device_for_model                          \\'
              '\n--dump\n'
              + '-' * 62 +
              '\nFor check inference against dumped results run:'
              '\npython3 cross_check_tool.py                        \\'
              '\n--input path/to/file/describing/input              \\'
              '\n--model path/to/model/*.xml                        \\'
              '\n--device device_for_model                          \\'
              '\n--load path/to/dump/file/* \n'
              + '-' * 62 +
              '\nFor all layers check provide:\n'
              '--layers=\'all\' \n'
              'For specific number of layers check provide:\n'
              '--layers=\'layer_name,another_layer_name,...,last_layer_name\'\n'
              + '-' * 62 +
              '\nIf --input is empty CCT generates input(s) from normal\n'
              'distribution and dumps this input to a file\n'
              + '-' * 62
    )

    model = parser.add_argument_group('Model specific arguments')
    model.add_argument('--input', '-i', type=str,
                       help='Path to an input image file or multi-input file to infer. Generates input(s) from normal '
                            'distribution if empty')
    # model.add_argument('--batch', '-b', type=int, help='Overrides batch size. Default is inherited from model')
    model.add_argument('--model', '-m', type=str, action=ExistingFileAction,
                       help='Path to an .xml file that represents the first IR of the trained model to infer.')
    model.add_argument('--reference_model', '-ref_m', type=str, action=ExistingFileAction,
                       help='Path to an .xml file that represents the second IR to compare the metrics. '
                            'Uses --model if empty')
    # TODO: allow to pass layer type
    model.add_argument('--layers', '-layers', type=str, default=None,
                       help='Defines layers to check. Options: all, None - for output layers check, list of '
                            'comma-separated layer names to check. Default value is None.')
    model.add_argument('-ref_layers', '--reference_layers', type=str, default=None,
                       help='Defines layers to check in referece model. Options: all, None - for output layers check, list of '
                            'comma-separated layer names to check. If not specified the same layers will be processed as in --layers parameter.')
    plugin = parser.add_argument_group('Plugin specific arguments')
    plugin.add_argument('--plugin_path', '-pp', type=str, action=ExistingDirAction, help='Path to a plugin folder.')
    plugin.add_argument('--device', '-d', type=str, required=True,
                        help='The first target device to infer the model specified with the -m or --model option. '
                             'CPU, GPU, HDDL or MYRIAD are acceptable.')
    plugin.add_argument('--config', '-conf', type=str, action=ExistingFileAction,
                        help='Path to config file for -d or -device device plugin')
    plugin.add_argument('--reference_device', '-ref_d', type=str,
                        help='The second target device to infer the model and compare the metrics. '
                             'CPU, GPU, HDDL or MYRIAD are acceptable.')
    plugin.add_argument('--reference_config', '-ref_conf', type=str, action=ExistingFileAction,
                        help='Path to config file for -ref_d or -reference_device device plugin')
    plugin.add_argument('-l', type=str, action=ExistingFileAction,
                        help='Required for (CPU)-targeted custom layers. Comma separated paths to a shared'
                             ' libraries with the kernels implementation.')

    modes = parser.add_argument_group('CCT mode arguments')
    # TODO eps? nobody uses it
    modes.add_argument('--dump', help='Enables tensors statistics dumping', action='store_true', default=False)
    modes.add_argument('--load', type=str, action=ExistingFileAction, help='Path to a file to load tensors from')
    model.add_argument('--num_of_iterations', '-ni', type=int, default=50,
                       help='Number of iterations to collect all over the net performance')
    parser.add_argument('-v', '--verbosity', action='store_true', default=False,
                        help='Increase output verbosity')
    return parser


@error_handling('validating arguments passed to cross_check_tool.py')
def validate_args(args):
    # input check
    if args.input is None:
        log.info('No input was provided by --input/-i. Generate input from noise')
    # model check
    if args.model is None and args.reference_model is None:
        raise Exception(
            "Parameters --model/-m and --reference_model/-ref_m are empty. At least one of them is required")
    elif args.model is None and args.reference_model:
        args.model = args.reference_model
    if args.model == args.reference_model:
        args.reference_model = None
    if args.model != args.reference_model and args.reference_model is not None and (args.layers is None or \
            args.reference_layers is None):
        log.warning('Check over two different IRs was enabled. In case if layer names in these two IRs are different, '
                    'please provide both -layers and --reference_layers to compare against.')
    # device check
    if args.device is None and args.reference_device is None:
        raise Exception("Parameters -device/-d and -reference_device/-ref_d are not set. Can not proceed."
                        "\nFor more details use -h option")
    if args.reference_device is None and args.reference_model is None and not args.dump and args.load is None:
        raise Exception("Please provide --reference_model/-ref_m to compare executions on different devices."
                        "\nAnother option is to provide --dump key to dump all execution info on one device."
                        "\nOr provide --load key to compare execution on device with dumped info"
                        "\nFor more details use -h option")
    if args.device is None:
        args.device = args.reference_device
        args.reference_device = None
    # dump and load check
    if args.dump and args.load is not None:
        raise Exception("Cross Check Tool does not support both loading and dumping modes to be enabled. "
                        "Choose one of them and proceed")
    if args.model is not None and args.reference_model is not None and args.dump or \
            args.device is not None and args.reference_device is not None and args.dump:
        raise Exception("Cross Check Tool does support dumping mode to be enabled only for one model on one device"
                        "\nFor more details use -h option")
    if args.model is not None and args.reference_model is not None and args.load is not None or \
            args.device is not None and args.reference_device is not None and args.load is not None:
        raise Exception("Cross Check Tool does support loading mode to be enabled for one model on one device against a"
                        " dumped file\nFor more details use -h option")
    return args


def find_out_cct_mode(args):
    """
    1 -- one IR mode
    2 -- two IRs mode
    3 -- dump mode
    4 -- load mode
    """
    # dump mode
    if args.dump and args.model is not None and args.device is not None and \
            args.reference_model is None and args.reference_device is None:
        return 3
    # load mode
    if args.load is not None and args.model is not None and args.device is not None and args.reference_device is None:
        return 4
    # two IR mode
    if args.model is not None and args.reference_model is not None:
        return 2
    # one IR mode
    if args.model is not None and args.reference_model is None:
        return 1
    raise Exception('Unknown Cross Check Tool CLI configuration.\nFor more details use -h option')


def print_inputs(inputs: list):
    word = 'inputs' if len(inputs) > 1 else 'input'
    log.info(f"{len(inputs)} {word} detected: {', '.join(input.any_name for input in inputs)}")

def print_output_ops(output_ops: list):
    layers = 'layers' if len(output_ops) > 1 else 'layer'
    log.info(f"Statistics will be dumped for {len(output_ops)} {layers}: {', '.join(op.friendly_name for op in output_ops)}")


###
#   PLUGIN
###


@error_handling('parsing config file for plugin: \'{config_file}\'')
def get_config_dictionary(config_file):
    config = {'PERF_COUNT': 'YES'}
    if not config_file:
        return config
    with open(config_file) as f:
        config_line = f.readline()
        key = config_line.split()[0]
        value = config_line[len(key):].strip()
        config[key] = value
    return config


###
#   INPUTS
###


def read_multi_input_file(input_file: str, model_inputs: list):
    npz = np.load(input_file, allow_pickle=True)
    files = npz.files
    dump = []
    for model_input in model_inputs:
        if model_input.any_name not in files:
            raise Exception(f"Can not find input data for input {model_input.any_name} in multi-input file {input_file}.\n"
                            f"Input data was provided for layers: {', '.join(files)}\n"
                            f"Network inputs: {', '.join(input.any_name for input in model_inputs)}")
        if 'tensor' in npz[model_input.any_name].item(0):
            just_tensor = npz[model_input.any_name].item(0)['tensor']
            input_shape = list(model_input.shape)
            log.info(f'Layer {model_input.any_name} shape = {input_shape}, input tensor from multi-input file shape = {just_tensor.shape}')
            try:
                reshaped_tensor = np.reshape(just_tensor, input_shape)
            except:
                raise Exception(f'Can not reshape input tensor from multi-input file for layer {model_input.any_name} to shape {input_shape}')
            dump.append(reshaped_tensor)
        else:
            raise Exception(
                f'Can not find \'tensor\' parameter for input {model_input.any_name} in input file {input_file}')
    return dump


def is_chw_input(input):
    if input.node.layout == Layout("NCHW") or input.node.layout == Layout("CHW"):
        return True
    shape = input.shape
    if len(shape) == 4:
        return shape[1] == 3
    if len(shape) == 3:
        return shape[0] == 3
    return False


def get_input_sizes(input):
    if is_chw_input(input):
        shape = input.shape
        if len(shape) == 3:
            return shape[1], shape[2]
        else:
            return shape[2], shape[3]
    else:
        shape = input.shape
        if len(shape) < 3 or len(shape) > 4:
            raise Exception('Can not interpret input shape as image')
        if len(shape) == 3:
            return shape[0], shape[1]
        else:
            return shape[1], shape[2]


@error_handling('reading --input/-i by OpenCV python module. OpenCV version: {}. '
                'It may happen due to wrong input image format'.format(cv2.__version__))
def read_image_file(image_file: str, model_input: Output):
    image_file = str(image_file)
    log.info(f'Prepare image {image_file}')
    image = cv2.imread(image_file)
    if image is None:
        raise Exception('Can not read input image ' + image_file)
    h, w = get_input_sizes(model_input)
    image = cv2.resize(image, (w, h))
    if is_chw_input(model_input):
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    if len(model_input.shape) == 4:
        image = np.expand_dims(image, 0) # Add batch dimension
    return image


def get_random_inputs(model_inputs, model_path):
    inputs = [np.clip(np.random.normal(0.5, 0.1, size=list(input.shape)), 0, 1) for input in model_inputs]
    dump_output_file(model_path + '_random_input_dump.npz', {model_inputs[i].any_name: {'tensor': inputs[i]} for i in range(len(model_inputs))})
    return inputs


def read_binary_file(bin_file, model_input):
    log.info(f"Prepare binary file {str(bin_file)}")
    binary_file_size = os.path.getsize(bin_file)
    tensor_size = model_input.tensor.size
    if tensor_size != binary_file_size:
        raise Exception(f"File {bin_file} contains {binary_file_size} bytes but model expects {tensor_size}")
    return np.reshape(np.fromfile(bin_file, get_dtype(model_input.element_type)), list(model_input.shape))


def input_processing(model_path: str, model_inputs: list, input_file: str):
    if input_file is None:
        return get_random_inputs(model_inputs, model_path)
    else:
        inputs = input_file.split(',')
        input_names = [input.any_name for input in model_inputs]
        input_data = {}
        for i in range(min(len(inputs), len(model_inputs))):
            splited = inputs[i].rsplit(':', maxsplit=1)
            if len(splited) == 0:
                raise Exception(f"Can't parse {'input_file'} input parameter!")
            tensor_name = None
            if len(splited) == 1:
                tensor_name = input_names[i]
            else:
                tensor_name = splited.pop(0)
            if tensor_name not in input_names:
                raise Exception(f"Input with name {tensor_name} doesn't exist in the model!")
            path = Path(splited.pop())
            if path.exists() and path.is_file():
                IMAGE_EXTENSIONS = ['.jpeg', '.jpg', '.png', '.bmp']
                BINARY_EXTENSIONS = ['.bin']
                NUMPY_EXTENSIONS = ['.npy', '.npz']
                current_input = model_inputs[input_names.index(tensor_name)]
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    input_data[tensor_name] = read_image_file(path, current_input)
                elif path.suffix.lower() in BINARY_EXTENSIONS:
                    input_data[tensor_name] = read_binary_file(path, current_input)
                elif path.suffix.lower() in NUMPY_EXTENSIONS:
                    return read_multi_input_file(path, model_inputs)
            elif path.is_dir():
                raise Exception(f"Path `{path}` is a directory! Provide full path to an input file!")
            elif not path.exists():
                raise Exception(f"Input file `{path}` doesn't exist!")
    return input_data


def accuracy_metrics(tensor, ref_tensor):
    if tensor.size != ref_tensor.size:
        raise Exception(f'Different number of elements in tensors {tensor.size} and {ref_tensor.size}. Can not compare')
    abs_diff = np.absolute(tensor - ref_tensor)
    np.seterr(divide='ignore', invalid='ignore')
    rel_diff = np.divide(abs_diff, np.min(abs_diff) if np.min(abs_diff) != 0 else 1e-20)

    metrics = [
        ('Max absolute difference', np.max(abs_diff)),
        ('Min absolute difference', np.min(abs_diff)),
        ('Max relative difference', np.max(rel_diff)),
        ('Min relative difference', np.min(rel_diff)),
        ('Min reference value', np.min(ref_tensor)),
        ('Min absolute reference value', np.min(np.abs(ref_tensor))),
        ('Max reference value', np.max(ref_tensor)),
        ('Max absolute reference value', np.max(np.abs(ref_tensor))),
        ('Min actual value', np.min(tensor)),
        ('Min absolute actual value', np.min(np.abs(tensor))),
        ('Max actual value', np.max(tensor)),
        ('Max absolute actual value', np.max(np.abs(tensor)))
    ]

    for key, value in metrics:
        if len(str(value)) > 5:
            log.info(f'{key:>35} : {value:.5E}', extra={'no_lvl': True})
        else:
            log.info(f'{key:>35} : {value}', extra={'no_lvl': True})
    return {metric: value for metric, value in metrics}


def performance_metrics(device, pc, ref_device, ref_pc):
    compare = [
        ('Device', '-d ' + device, '-ref_d ' + ref_device),
        ('Status', str(pc.status), str(ref_pc.status)),
        ('Layer type', pc.node_type, ref_pc.node_type),
        ('Real time, microsec', str(pc.real_time), str(ref_pc.real_time))
    ]

    for metric, actual, reference in compare:
        log.info(f'{metric:>35}: {actual:>16} {reference:>16}', extra={'no_lvl': True})


def tensor_counters(tensor, ref_tensor):
    counters = [
        ('Number of NAN', np.sum(np.isnan(tensor)), np.sum(np.isnan(ref_tensor))),
        ('Number of INF', np.sum(np.isinf(tensor)), np.sum(np.isinf(ref_tensor))),
        ('Number of ZERO', tensor.size - np.count_nonzero(tensor),
         ref_tensor.size - np.count_nonzero(ref_tensor))
    ]
    for metric, actual, reference in counters:
        log.info(f'{metric:>35}: {actual:>16} {reference:>16}', extra={'no_lvl': True})


def update_global_accuracy_matrics(global_accuracy: list, current_accuracy: dict):
    metrics = [
        ('Max absolute difference', lambda x, y: max(x, y)),
        ('Min absolute difference', lambda x, y: min(x, y)),
        ('Max relative difference', lambda x, y: max(x, y)),
        ('Min relative difference', lambda x, y: min(x, y))]
    for metric, formula in metrics:
        global_metric = [item for item in global_accuracy if item[0] == metric]
        if len(global_metric) == 1:
            g_metric, g_value = global_metric[0]
            global_accuracy.remove(global_metric[0])
            global_accuracy.append((metric, formula(g_value, current_accuracy[metric])))
        else:
            global_accuracy.append((metric, current_accuracy[metric]))
    return global_accuracy


def print_all_over_the_net_metrics(global_accuracy: list, global_times: list = None,
                                   ref_global_times: list = None):
    if global_times is not None and ref_global_times is not None and len(global_times) and len(ref_global_times):
        log.info('-' * 70, extra={'no_lvl': True})
        log.info(f'{"Overall performance, microseconds":>35}: '
                                f'{global_times[len(global_times) // 2].microseconds:>16,.5E} '
                                f'{ref_global_times[len(ref_global_times) // 2].microseconds:>16,.5E}', 
                                                                                extra={'no_lvl': True})
        log.info('-' * 70, extra={'no_lvl': True})
    for metric, value in global_accuracy:
        log.info(f"Overall {metric.lower()} = {value}")


###
#   MAPPING
###


def get_ops_list(all_ops: list, outputs: list, layers: str):
    if layers is not None and layers != 'None':
        if layers == 'all':
            return [op for op in all_ops if op.get_type_name() not in ['Constant', 'Result', 'Parameter']]
        else:
            all_ops_map = {op.friendly_name: op for op in all_ops}
            user_ops = [layer.strip() for layer in layers.split(',')]
            new_outputs = []
            for user_op in user_ops:
                if user_op not in all_ops_map:
                    raise Exception(f"Operation with name `{user_op}` doesn't exist in the model")
                else:
                    new_outputs.append(all_ops_map[user_op])
            return new_outputs
    else:
        return [output.node for output in outputs]


def perf_counts_to_dump(prof_info):
    perf_count = {}
    perf_count["status"] = prof_info.status
    perf_count["real_time"] = prof_info.real_time
    perf_count["cpu_time"] = prof_info.cpu_time
    perf_count["exec_type"] = prof_info.exec_type
    perf_count["node_type"] = prof_info.node_type
    return perf_count


def load_profiling_info(pi_dumped):
    prof_info = ProfilingInfo()
    for property, value in pi_dumped.items():
        setattr(prof_info, property, value)
    return prof_info


###
#   FILES
###

def dump_output_file(output_file, dump_dict):
    np.savez_compressed(output_file, **dump_dict)
    log.info(f'Dump file path: {output_file}')


def load_dump(file_to_load: str):
    npz = np.load(file_to_load, allow_pickle=True)
    dump = {file: [npz[file].item(i).item(0)  for i in range(npz[file].size)] for file in npz if file != "device"}
    dump["device"] = npz["device"].item(0)
    return dump
