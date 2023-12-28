import logging as log
import subprocess
import sys
from typing import Union
import tensorflow as tf


import numpy as np
from openvino.runtime import Dimension, PartialShape


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
