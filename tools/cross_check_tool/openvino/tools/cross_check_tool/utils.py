# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log
import os
import sys
import traceback
import xml

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
              '\nFor dumping blob and performance counters run:'
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
    model.add_argument('--input', '-i', type=str, action=ExistingFileAction,
                       help='Path to an input image file or multi-input file to infer. Generates input(s) from normal '
                            'distribution if empty')
    # model.add_argument('--batch', '-b', type=int, help='Overrides batch size. Default is inherited from model')
    model.add_argument('--model', '-m', type=str, action=ExistingFileAction,
                       help='Path to an .xml file that represents the first IR of the trained model to infer.')
    model.add_argument('--reference_model', '-ref_m', type=str, action=ExistingFileAction,
                       help='Path to an .xml file that represents the second IR to compare the metrics. '
                            'Uses --model if empty')
    model.add_argument('--layers', '-layers', type=str, default=None,
                       help='Defines layers to check. Options: all, None - for output layers check, list of '
                            'comma-separated layer names to check. Default value is None.')
    model.add_argument('--mapping', '-map', type=str, action=ExistingFileAction,
                       help='Model Optimizer provided mapping for --model/-m')
    model.add_argument('--reference_mapping', '-ref_map', type=str, action=ExistingFileAction,
                       help='Model Optimizer provided mapping for --reference_model/-ref_model')

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
                        help='Required for MKLDNN (CPU)-targeted custom layers. Comma separated paths to a shared'
                             ' libraries with the kernels implementation.')

    modes = parser.add_argument_group('CCT mode arguments')
    # TODO eps? nobody uses it
    modes.add_argument('--dump', help='Enables blobs statistics dumping', action='store_true', default=False)
    modes.add_argument('--load', type=str, action=ExistingFileAction, help='Path to a file to load blobs from')
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
    if args.model != args.reference_model and args.reference_model is not None and args.mapping is None and \
            args.reference_mapping is None:
        log.warning('Check over two different IRs was enabled. In case if layer names in this two IRs differ, '
                    'please provide mapping files with --mapping/-map and --reference_mapping/-ref_map')
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


def print_input_layers(inputs: list):
    word = 'inputs' if len(inputs) > 1 else 'input'
    log.info(f"{len(inputs)} {word} detected: {', '.join(inputs)}")


def print_output_layers(outputs: list):
    layers = 'layers' if len(outputs) > 1 else 'layer'
    log.info(f"Statistics will be dumped for {len(outputs)} {layers}: {', '.join(outputs)}")


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


def read_multi_input_file(input_file: str, net_inputs: dict):
    npz = np.load(input_file, allow_pickle=True)
    files = npz.files
    dump = {}
    for net_input in net_inputs:
        if net_input not in files:
            raise Exception(f"Can not find input data for input {net_input} in multi-input file {input_file}.\n"
                            f"Input data was provided for layers: {', '.join(files)}\n"
                            f"Network inputs: {', '.join(net_inputs.keys())}")
        if 'blob' in npz[net_input].item(0):
            just_blob = npz[net_input].item(0)['blob']
            network_shape = net_inputs[net_input].input_data.shape
            log.info(f'Layer {net_input} shape = {network_shape}, input blob from multi-input file shape = {just_blob.shape}')
            try:
                reshaped_blob = np.reshape(just_blob, network_shape)
            except:
                raise Exception(f'Can not reshape input blob from multi-input file for layer {net_input} to shape {network_shape}')
            dump[net_input] = reshaped_blob
        else:
            raise Exception(
                f'Can not find \'blob\' parameter for input {net_input} in input file {input_file}')
    return dump


@error_handling('reading --input/-i by OpenCV python module. OpenCV version: {}. '
                'It may happen due to wrong input image format'.format(cv2.__version__))
def read_image_file(input_file: str, net_inputs: dict):
    inputs = dict()
    if len(net_inputs) == 1:
        image = cv2.imread(input_file)
        if image is None:
            raise Exception('Can not read input image ' + input_file)
        only_layer_name = list(net_inputs.keys())[0]
        shape = net_inputs[only_layer_name].input_data.shape
        if len(shape) != 4:
            raise Exception('Can not interpret input shape as image')
        n, c, h, w = shape
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        image = image.reshape((n, c, h, w))
        inputs[only_layer_name] = image
    else:
        raise Exception('Multi-input topology detected. Please provide multi-input file to --input key')
    return inputs


def input_processing(model_path: str, net_inputs: dict, input_file: str, layers_map: dict = None):
    inputs = dict()
    if input_file is None:
        for net_input in net_inputs:
            inputs[net_input] = np.clip(np.random.normal(0.5, 0.1, size=net_inputs[net_input].input_data.shape), 0, 1)
        dump_output_file(model_path + '_random_input_dump.npz', {inp: {'blob': inputs[inp]} for inp in inputs})
        return inputs
    try:
        inputs = read_multi_input_file(input_file=input_file, net_inputs=net_inputs)
    except:
        inputs = read_image_file(input_file=input_file, net_inputs=net_inputs)
    return inputs


def accuracy_metrics(out_blob, ref_out_blob):
    if out_blob.size != ref_out_blob.size:
        raise Exception(f'Different number of elements in blobs {out_blob.size} and {ref_out_blob.size}. Can not compare')
    abs_diff = np.absolute(out_blob - ref_out_blob)
    rel_diff = np.divide(abs_diff, np.min(abs_diff) if np.min(abs_diff) != 0 else 1e-20)

    metrics = [
        ('Max absolute difference', np.max(abs_diff)),
        ('Min absolute difference', np.min(abs_diff)),
        ('Max relative difference', np.max(rel_diff)),
        ('Min relative difference', np.min(rel_diff)),
        ('Min reference value', np.min(ref_out_blob)),
        ('Min absolute reference value', np.min(np.abs(ref_out_blob))),
        ('Max reference value', np.max(ref_out_blob)),
        ('Max absolute reference value', np.max(np.abs(ref_out_blob))),
        ('Min actual value', np.min(out_blob)),
        ('Min absolute actual value', np.min(np.abs(out_blob))),
        ('Max actual value', np.max(out_blob)),
        ('Max absolute actual value', np.max(np.abs(out_blob)))
    ]

    for key, value in metrics:
        if len(str(value)) > 5:
            log.info(f'{key:>35} : {value:.5E}', extra={'no_lvl': True})
        else:
            log.info(f'{key:>35} : {value}', extra={'no_lvl': True})
    return {metric: value for metric, value in metrics}


def performance_metrics(pc, ref_pc):
    compare = [
        ('Device', '-d ' + pc['device'], '-ref_d ' + ref_pc['device']),
        ('Status', pc['status'], ref_pc['status']),
        ('Layer type', pc['layer_type'], ref_pc['layer_type']),
        ('Real time, microsec', pc['real_time'], ref_pc['real_time'])
    ]

    for metric, actual, reference in compare:
        log.info(f'{metric:>35}: {actual:>16} {reference:>16}', extra={'no_lvl': True})


def blob_counters(out_blob, ref_out_blob):
    counters = [
        ('Number of NAN', np.sum(np.isnan(out_blob)), np.sum(np.isnan(ref_out_blob))),
        ('Number of INF', np.sum(np.isinf(out_blob)), np.sum(np.isinf(ref_out_blob))),
        ('Number of ZERO', out_blob.size - np.count_nonzero(out_blob),
         ref_out_blob.size - np.count_nonzero(ref_out_blob))
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


def print_all_over_the_net_metrics(global_accuracy: (str, float), global_times: list = None,
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


def read_mapping(file_name: str):
    # TODO check twice
    mapping_dict = {}
    xml_tree = xml.etree.ElementTree.parse(file_name)
    xml_root = xml_tree.getroot()
    for child in xml_root:
        fw_info = child.find('.//framework')
        ir_info = child.find('.//IR')
        if fw_info is None:
            continue
        if ir_info is None:
            continue
        framework_name = fw_info.attrib['name'] + ':' + fw_info.attrib['out_port_id']
        ir_name = ir_info.attrib['name'] if ir_info is not None else None
        ir_layer_id = int(ir_info.attrib['id']) if ir_info is not None else None
        mapping_dict[framework_name] = (ir_name, ir_layer_id)
    return mapping_dict


def map_layers(mapping_file: str = None, ref_mapping_file: str = None):
    if mapping_file is not None and ref_mapping_file is not None:
        mapping = read_mapping(mapping_file)
        ref_mapping = read_mapping(ref_mapping_file)
        mapping = {layer: ref_layer for layer in mapping for ref_layer in ref_mapping if layer == ref_layer}
        return mapping


def manage_user_outputs_with_mapping(mapping, reference_mapping, user_layers):
    if mapping is not None and reference_mapping is not None:
        layers_map = map_layers(mapping, reference_mapping)
    else:
        layers_map = {layer: layer for layer in user_layers}
    for layer in user_layers:
        if layer not in layers_map:
            if mapping is not None and reference_mapping is not None:
                log.warning(
                    f'Can not map layer {layer} from --model/-m to any layer from --reference_model/-ref_m')
            else:
                log.warning(f'Can not find layer {layer} in --reference_model/-ref_m model')
    for layer in layers_map:
        if layer not in user_layers:
            del layers_map[layer]
    return layers_map


def get_layers_list(all_layers: list, inputs: dict, outputs: list, layers: str):
    if layers is not None and layers != 'None':
        if layers == 'all':
            return {layer.get_friendly_name(): layer for layer in all_layers \
                    if layer.get_type_name() not in ['Constant', 'Result']}
        else:
            all_layers_names = {op.get_friendly_name() : op for op in all_layers}
            user_layers = [layer.strip() for layer in layers.split(',')]
            layers_to_check = []
            for user_layer in user_layers:
                if user_layer not in all_layers_names:
                    raise Exception(f"Layer {user_layer} doesn't exist in the model")
                if user_layer in inputs:
                    raise Exception(f"Layer {user_layer} is input layer. Can not proceed")
                if all_layers_names[user_layer].get_type_name() != 'Result':
                    layers_to_check.append(user_layer)
                else:
                    # if layer type is Result - add previous layer
                    prev_layer = all_layers[len(all_layers)-2]
                    layers_to_check.append(prev_layer.get_friendly_name())
            return layers_to_check
    else:
        return outputs


###
#   FILES
###

def dump_output_file(output_file, dump_dict):
    np.savez_compressed(output_file, **dump_dict)
    log.info(f'Dump file path: {output_file}')


def load_dump(file_to_load: str):
    npz = np.load(file_to_load, allow_pickle=True)
    dump = {file: npz[file].item(0) for file in npz}
    return dump
