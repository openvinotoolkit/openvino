# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import logging as log
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Union
from filelock import FileLock

from contextlib import contextmanager
from datetime import datetime
import copy
import numpy as np
import tensorflow as tf
import subprocess


from openvino.runtime import Dimension, PartialShape


log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)


_csv_bool_map = {"true": True, "false": False,
                 True: "true", False: "false"}

PRECISION_MAP = {
    'FP32': 'f32',
    'BF16': 'bf16'
}


def write_to_csv(csv_path: Path, data: list):
    """
    Writes specified data to a CSV-formatted file
    :param csv_path: path to CSV-formatted file to write data
    :param data: data to write
    :return: None
    """
    # NOTE: concurrent writing to a file using `csv` module may lead
    # to lost of several rows, but every row isn't corrupted.
    # In case of using it for IRs pre-generation (in `collect_irs.py`),
    # `test.py` is ready that some records may be not available.
    with open(str(csv_path), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(data)


def write_irs_mapping_file(path: Path, ir_tag: str, status: bool, mo_log: [str, Path, None],
                           xml: [str, Path, None], bin: [str, Path, None], timeout: int = 30):
    """
    Writes record to IRs mapping file.
    IR related paths are saved in relative format to support different
    OS specific base paths.

    :param path: path to CSV-formatted IRs mapping file
    :param ir_tag: tag to map IRs
    :param status: status of IRs pre-generation
    :param mo_log: full path to MO log file
    :param xml: full path to IR's .xml file
    :param bin: full path to IR's .bin file,
    :param timeout: filelock timeout in seconds
    :return: dictionary with IRs mapping
    """
    assert path.parent.exists(), "File's parent directory should exists"

    def _rel_path(path, parent_path):
        return Path(path).relative_to(parent_path) if path is not None else None

    status = _csv_bool_map[status]
    mo_log, xml, bin = _rel_path(mo_log, path.parent), _rel_path(xml, path.parent), _rel_path(bin, path.parent)
    log.info("Prepare record to a mapping file: {}".format({ir_tag: [status, mo_log, xml, bin]}))

    lock_irs_mapping_path = path.with_suffix('.lock')

    with FileLock(lock_irs_mapping_path, timeout):
        write_to_csv(path, [ir_tag, status, mo_log, xml, bin])


def read_irs_mapping_file(path: Path, timeout: int = 30, lock_access: bool = False):
    """
    Reads IRs mapping file
    :param path: path to CSV-formatted IRs mapping
    :param timeout: filelock timeout in seconds
    :param lock_access: boolean which specifies should the file be locked or not.
    Lock is required when read/write simultaneously in parallel.
    :return: dictionary with IRs mapping
    """
    def _full_path(path, parent_path):
        # `csv` module converts None to empty string, so implicitly convert it to None
        return parent_path / path if path else None

    def _read(csv_path):
        with open(str(csv_path), 'r', newline='') as csvfile:
            fixed_csvfile = (line.replace('\0', '') for line in
                             csvfile)  # replace '\0' to prevent "_csv.Error: line contains NULL byte"
            reader = csv.reader(fixed_csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            irs_mapping = {}
            for row in reader:
                if row:
                    try:
                        ir_tag, status, rel_mo_log, rel_xml, rel_bin = row
                        status = _csv_bool_map[status]
                        mo_log, xml, bin = _full_path(rel_mo_log, path.parent), _full_path(rel_xml, path.parent), \
                                           _full_path(rel_bin, path.parent)
                        irs_mapping.update({ir_tag: [status, mo_log, xml, bin]})
                    except:
                        pass  # ignore any corrupted row
        return irs_mapping

    lock_irs_mapping_path = path.with_suffix('.lock')

    if not lock_access:
        irs_mapping = _read(path)
    else:
        with FileLock(lock_irs_mapping_path, timeout):
            irs_mapping = _read(path)

    return irs_mapping


def get_ir_tag(name, ir_version, precision, batch, sequence_length=None, skip_mo_args=None):
    """
    Prepares tag to map IR generated in E2E test
    :param name: test (or any specific) name
    :param ir_version: version of IR (e.g. v11)
    :param precision: precision value of IR
    :param batch: batch value of IR
    :param sequence_length: sequence_length value of IR
    :param skip_mo_args: line with comma separated args that will be deleted from MO cmd line

    :return: IR tag
    """
    model_tag = f"{name}_IR_{ir_version}_{precision}_batch_{batch}"
    if sequence_length:
        model_tag = f"{model_tag}_seqlen_{sequence_length}"
    if skip_mo_args:
        model_tag += f"deleted_mo_args:_{skip_mo_args}"
    return model_tag


def store_data_to_csv(csv_path, instance, ir_version, data, device, data_name, skip_mo_args=None):
    """
    This function helps to store runtime data such as time of IR generation by MO or of network loading to plugin.
    To store it, please, execute test.py (both for MO and load net to plugin) with keys:
    `--ir_gen_time_csv_name <name_of_csv>` and `--load_net_to_plug_time_csv_name <name_of_csv>`

    :param csv_path: csv_path for saving csv file
    :param instance: test class instance
    :param ir_version: string in format "vN" (e.g. v11)
    :param device: in general, it is device for execution, but for MO it is useless. But for MO it is set as 'CPU'
    :param data_name: name of runtime data (such as attribute where time was saved)
    :param skip_mo_args: line with comma separated args that will be deleted from MO cmd line
    :return:
    """
    csv_header = ["Model Tag", "Device", "Data", "Operation type"]
    if not os.path.exists(csv_path):
        write_to_csv(csv_path=csv_path, data=csv_header)
    model_mapping_tag = get_ir_tag(instance.__class__.__name__, ir_version, instance.precision,
                                   instance.batch, instance.required_params.get("sequence_length", None),
                                   skip_mo_args)
    write_to_csv(csv_path=csv_path, data=[model_mapping_tag, device, data, data_name])


def class_factory(cls_name, cls_kwargs, BaseClass):
    """
    Function that generates of custom classes
    :param cls_name: name of the future class
    :param cls_kwargs: attributes required for the class (e.g. __is_test_config__)
    :param BaseClass: basic class where implemented behaviour of the test
    :return:
    """

    # Generates new class with "cls_name" type inherited from "object" and
    # with specified "__init__" and other class attributes

    newclass = type(cls_name, (BaseClass,), {**cls_kwargs})
    return newclass


def remove_mo_args(mo_args_to_skip: Union[list, str], mo_cmd):
    """
    This function deletes arguments from MO cmd line

    :param mo_args_to_skip: mo arguments to delete
    :param mo_cmd: MO command line that is supposed to be reconfigured
    """
    mo_args_to_skip = mo_args_to_skip if isinstance(mo_args_to_skip, list) else mo_args_to_skip.split(',')

    for mo_arg in mo_args_to_skip:
        if mo_arg in mo_cmd:
            log.info('Deleting argument from MO cmd: {}'.format(mo_arg))
            del mo_cmd[mo_arg]

    return mo_cmd


def remove_mo_args_oob(mo_args_to_skip: Union[list, str], mo_cmd: dict, instance) -> dict:
    """
    This function removes shapes from MO cmd line of instance
    If test instance has specific inputs for MO cmd then
    "--input" will be equal to it

    :param mo_args_to_skip: mo arguments to delete
    :param mo_cmd: MO command line that is supposed to be reconfigured
    :param instance: test instance
    """

    mo_input = mo_cmd.get('input')
    mo_cmd = remove_mo_args(mo_args_to_skip, mo_cmd)

    if mo_input and hasattr(instance, 'frozen_inputs'):
        mo_cmd['input'] = instance.frozen_inputs

    return mo_cmd


def get_framework_from_model_ex(path_to_test_file):
    frameworks_path = {'caffe': 'caffe',
                       'kaldi': 'kaldi',
                       'mxnet': 'mxnet',
                       'onnx': 'onnx',
                       'paddlepaddle': 'paddle',
                       'pytorch': 'pytorch',
                       'tf': 'tf',
                       'tflite': 'tflite',
                       'tf_2x': 'tf2'}
    pattern = r'pipelines\w*[\\/]\w+[\\/](\w+)[\\/]'
    name_fw = re.search(pattern, path_to_test_file)
    if name_fw:
        return frameworks_path.get(name_fw.group(1), 'Undefined')

    return 'Undefined'


def align_output_name(name, outputs):
    if isinstance(name, int):
        return name
    if ":" in name:
        name_without_port, port = name.rsplit(":", 1)
        use_name_without_port = name_without_port in outputs and port.isnumeric()
        if use_name_without_port:
            return use_name_without_port, name_without_port
    name_with_default_port = name + ":0"
    if name_with_default_port in outputs:
        return name_with_default_port


def construct_names_set(name):
    if ":" in name:
        name_without_port, port = name.rsplit(":", 1)
        if port.isnumeric():
            return {name_without_port, name}
    name_with_default_port = name + ":0"
    return {name_with_default_port, name}


def align_input_names(input_dict, model):
    if all([isinstance(x, int) for x in input_dict]):
        return input_dict
    new_input_dict = {}
    for input_data_layer in input_dict:
        new_input_dict[input_data_layer] = input_dict[input_data_layer]
        for input_layer in model.inputs:
            common_names = input_layer.names.intersection(construct_names_set(input_data_layer))
            if common_names:
                if input_data_layer not in common_names:
                    new_input_dict[common_names.pop()] = new_input_dict.pop(input_data_layer)
    return new_input_dict


def get_infer_result(input_data, compiled_model, ov_model, infer_run_counter=0, index_infer=False):
    log.info("Starting inference")
    log.info("Inference run counter: " + str(infer_run_counter + 1))

    request = compiled_model.create_infer_request()
    cur_input_data = align_input_names(input_data, ov_model)
    infer_result = request.infer(cur_input_data)

    helper = {}

    if index_infer:
        for i, out_tensor in enumerate(infer_result.values()):
            helper[i] = out_tensor
    else:
        for out_obj, out_tensor in infer_result.items():
            assert out_obj.names, "Output tensor doesn't have name"
            tensor_name = out_obj.get_any_name()
            if tensor_name in helper:
                tensor_name = next(iter(out_obj.names - set(helper.keys())), tensor_name)
            helper[tensor_name] = out_tensor

    return helper


def get_shapes_with_frame_size(default_shapes, ov_model, input_data):
    # there could be dynamic shapes in ov_model.inputs, therefore shapes should be known from test
    inputs = ov_model.inputs if default_shapes is None else default_shapes

    for input_layer in inputs:
        if default_shapes:
            frame_size = default_shapes[input_layer]
            input_data[input_layer] = input_data[input_layer].reshape(-1, *frame_size)
        else:
            layer_name = input_layer.names.intersection(set(input_data.keys())).pop()
            frame_size = [dim for dim in input_layer.shape]
            input_data[layer_name] = input_data[layer_name].reshape(-1, *frame_size)

    return input_data


def copy_files_by_pattern(directory: Path, pattern_to_find: str, pattern_to_copy: str):
    for file in directory.glob("{}*".format(pattern_to_find)):
        file_extension = ''.join(file.suffixes)
        copied_file = file.parent / (pattern_to_copy + file_extension)
        if file.exists():
            log.info('Copying file from {} to {}'.format(file, copied_file))
            try:
                shutil.copy(str(file), str(copied_file))
            except shutil.SameFileError:
                pass
        else:
            log.info('File {} does not exist'.format(file))


def check_mo_precision(instance):
    # SPR use BF16 precision by default, and it requires thresholds that are different from FP32 threshold
    # Run Model Optimizer with FP32 precision because it hasn't bf16 option
    if 'get_ovc_model' in instance['get_ir'] and instance['get_ir']['get_ovc_model']['precision'] == "BF16":
        log.info("Setting precision FP32 for Model Optimizer...")
        instance['get_ir']['get_ovc_model']['precision'] = "FP32"


def set_infer_precision_hint(instance, pipeline, inference_precision_hint):
    api = next(iter(pipeline.get('infer')))
    # inference_precision_hint is required only for GPU
    if instance.device == 'GPU':
        if inference_precision_hint:
            # f16 is default value
            supported_values = ['bf16', 'f32']
            assert inference_precision_hint in supported_values, f"{inference_precision_hint} not in" \
                                                                 f" supported values: {supported_values}"
            pipeline['infer'][api]['plugin_config'] = {
                'INFERENCE_PRECISION_HINT': inference_precision_hint}
        else:
            test_precision = instance.precision
            if test_precision != 'FP16':
                inference_precision_hint = PRECISION_MAP[test_precision]
                pipeline['infer'][api]['plugin_config'] = {
                    'INFERENCE_PRECISION_HINT': inference_precision_hint}

    return pipeline


class BrokenTestException(Exception):
    """
    Custom exception type required for catching only errors related to incorrectly defined test pipeline
    """
    pass


class BrokenTest:
    """
    Class which used to substitute the test pipeline class which are incorrectly defined.
    Used in conftest pytest plugins on test collection stage. If during creation of test pipeline instance
    some exception happens, the pipeline class replaced with BrokenTest class with keeping original class name.
    Attempt to refer to any test pipeline in test runners leads to raising original error happened in initial
    test pipeline class
    """

    def __init__(self, test_id, fail_message, exception, *args, **kwargs):
        """
        :param test_id: test identificator
        :param fail_message: string which should be logged while reference to ie_pipeline or ref_pipeline attributes
        :param exception: exception which will be raised while reference to ie_pipeline or ref_pipeline attributes
        :param args: auxiliary positional arguments
        :param kwargs: auxiliary keyword arguments
        """
        self.test_id = test_id
        self.fail_message = fail_message
        self.exception = exception

    @property
    def ref_pipeline(self, *args, **kwargs):
        log.error(self.fail_message)
        raise BrokenTestException(str(self.exception))

    @property
    def ie_pipeline(self, *args, **kwargs):
        log.error(self.fail_message)
        raise BrokenTestException(str(self.exception))

    @property
    def prepare_prerequisites(self, *args, **kwargs):
        log.error(self.fail_message)
        raise BrokenTestException(str(self.exception))


@contextmanager
def log_timestamp(action):
    """
    Function adds timestamp for the start and the end of the action 
    :param action: name of action for logging 
    """
    log.debug(f'{datetime.fromtimestamp(datetime.now().timestamp(), tz=None)}: Started {action}')
    yield 
    log.debug(f'{datetime.fromtimestamp(datetime.now().timestamp(), tz=None)}: Finished {action}')


def timestamp():
    """
    Function return current timestamp for logging
    """
    return f'{datetime.fromtimestamp(datetime.now().timestamp(), tz=None)}'


def get_static_shape(default_shapes, changed_values, layout, dims_to_change):
    static_shapes = copy.deepcopy(default_shapes)
    static_shapes = {k: list(v) for k, v in static_shapes.items()}
    for input_layer, dimension in dims_to_change.items():
        if dimension is None:
            continue
        else:
            dim_indexes = [layout[input_layer].index(d) for d in dimension]
            for value_index, value in enumerate(dict(changed_values)[input_layer]):
                if value is None:
                    static_shapes[input_layer][dim_indexes[value_index]] = \
                        default_shapes[input_layer][dim_indexes[value_index]]
                else:
                    static_shapes[input_layer][dim_indexes[value_index]] = value
    return static_shapes


def get_shapes_from_data(input_data) -> dict:
    shapes = {}
    for input_layer in input_data:
        shapes[input_layer] = PartialShape(input_data[input_layer].shape)
    return shapes


def convert_shapes_to_partial_shape(shapes: dict) -> dict:
    partial_shape = {}
    for layer, shape in shapes.items():
        dimension_tmp = []
        for item in shape:
            dimension_tmp.append(Dimension(item[0], item[1])) if type(item) == list else dimension_tmp.append(
                Dimension(item))
        partial_shape[layer] = PartialShape(dimension_tmp)
    return partial_shape


def name_aligner(infer_result, reference, xml=None):
    """
    Function name_aligner aligns names for inference and reference outputs if number of their outputs == 1
    """
    if len(infer_result.keys()) == 1 == len(reference.keys()):
        log.info("Renaming inferred output layer {} to referenced output layer {}".format(
            list(infer_result.keys())[0], list(reference.keys())[0]))
        infer_result[next(iter(reference))] = infer_result.pop(next(iter(infer_result)))

    return infer_result, reference


def shell(cmd, env=None, cwd=None, out_format="plain", log=True):
    """
    Run command execution in specified environment

    :param cmd: list containing command and its parameters
    :param env: set of environment variables to set for this command
    :param cwd: working directory from which execute call
    :param out_format: 'plain' or 'html'. If 'html' all '\n; symbols are replaced by '<br>' tag
    :param log: display output info into sys.stdout or not
    :return: returncode, stdout, stderr
    """
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        cmd = ['/bin/bash', '-c', "unset OMP_NUM_THREADS; " + " ".join(cmd)]
    else:
        cmd = " ".join(cmd)
    if log:
        sys.stdout.write("Running command:\n" + "".join(cmd) + "\n")
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    stdout = str(stdout.decode('utf-8'))
    stderr = str(stderr.decode('utf-8'))
    if out_format == "html":
        stdout = "<br>\n".join(stdout.split('\n'))
        stderr = "<br>\n".join(stderr.split('\n'))
    return p.returncode, stdout, stderr

