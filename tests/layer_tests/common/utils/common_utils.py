# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import numpy as np
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_ir(coverage=False, **kwargs):
    from openvino.tools.mo import mo
    mo_path = Path(mo.__file__).parent
    mo_runner = mo_path.joinpath('main.py').as_posix()
    if coverage:
        params = [sys.executable, '-m', 'coverage', 'run', '-p', '--source={}'.format(mo_path.parent),
                  '--omit=*_test.py', mo_runner]
    else:
        params = [sys.executable, mo_runner]
    for key, value in kwargs.items():
        if key == "batch":
            params.extend(("-b", str(value)))
        elif key == "k":
            params.extend(("-k", str(value)))
        # for FP32 set explicitly compress_to_fp16=False,
        # if we omit this argument for FP32, it will be set implicitly to True as the default
        elif key == 'compress_to_fp16':
            params.append("--{}={}".format(key, value))
        elif isinstance(value, bool) and value:
            params.append("--{}".format(key))
        elif isinstance(value, bool) and not value:
            continue
        elif (isinstance(value, tuple) and value) or (isinstance(value, str)):
            params.extend(("--{}".format(key), str('"{}"'.format(value))))
        elif key == "mean_values" and (' ' in value or '(' in value):
            params.extend(("--{}".format(key), str('"{}"'.format(value))))
        else:
            params.extend(("--{}".format(key), str(value)))
    exit_code, stdout, stderr = shell(params)
    logger.info("Model Optimizer out:\n{}".format(stdout))
    logger.error(stderr)
    return exit_code, stderr


def generate_ir_python_api(coverage=False, **kwargs):
    out_dir = kwargs['output_dir'] + os.sep + "model.xml"
    if 'use_legacy_frontend' in kwargs and kwargs['use_legacy_frontend']:
        from openvino.runtime import serialize
        from openvino.tools.mo import convert_model

        ov_model = convert_model(**kwargs)
        serialize(ov_model, out_dir)
    else:
        from openvino import convert_model, save_model

        # cleanup parameters for convert
        if 'output_dir' in kwargs:
            del kwargs['output_dir']

        compress_to_fp16 = False
        if 'compress_to_fp16' in kwargs:
            # TODO 132871: fix error with no tensor name in case of compression
            # compress_to_fp16 = kwargs['compress_to_fp16']
            del kwargs['compress_to_fp16']

        ov_model = convert_model(**kwargs)
        save_model(ov_model, out_dir, compress_to_fp16)

    return 0, ""


def generate_ir_ovc(input_model, opts):
    params = ['ovc', input_model]
    for key, value in opts.items():
        # handle optional arguments
        # both key and values are of string type
        params.extend(("--{}".format(key), value))
    exit_code, stdout, stderr = shell(params)
    return exit_code, stdout, stderr


def shell(cmd, env=None, cwd=None, out_format="plain"):
    """
    Run command execution in specified environment
    :param cmd: list containing command and its parameters
    :param env: set of environment variables to set for this command
    :param cwd: working directory from which execute call
    :param out_format: 'plain' or 'html'. If 'html' all '\n; symbols are replaced by '<br>' tag
    :return:
    """
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        cmd = ['/bin/bash', '-c', "unset OMP_NUM_THREADS; " + " ".join(cmd)]
    else:
        cmd = " ".join(cmd)

    sys.stdout.write("Running command:\n" + "".join(cmd) + "\n")
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    stdout = str(stdout.decode('utf-8'))
    stderr = str(stderr.decode('utf-8'))
    if out_format == "html":
        stdout = "<br>\n".join(stdout.split('\n'))
        stderr = "<br>\n".join(stderr.split('\n'))
    return p.returncode, stdout, stderr


def allclose(cur_array, ref_array, atol, rtol):
    """
    Comparison of abs_diff and rel_diff with tolerances for every values of corresponding elements.
    If (abs_diff < atol) or (rel_diff < rtol) for every element, comparison of elements will pass, else will fail.
    Note: if value is very small, firstly abs_diff will be used. If value is huge, abs_diff may be failed,
    and rel_diff will be used. So if tensor has small and huge values, need to compare every value
    with abs_diff and rel_diff instead of using one of it for the whole array.
    :param cur_array: tensor from IE
    :param ref_array: tensor from FW
    :param atol: absolute tolerance (threshold for absolute difference)
    :param rtol: relative tolerance (threshold for relative difference)
    :return: bool value means that values of tensors are equal with tolerance or not
    """
    try:
        # a scalar case of string type
        # need to repack TF output to python string
        if isinstance(ref_array, bytes):
            ref_array = np.array(str(ref_array, 'utf-8'))
    except:
        pass

    if cur_array.dtype.type == str or cur_array.dtype.type == np.str_:
        # TF can represent string tensors in different format: array of bytestreams
        # so we have to align formats of both string tensors, for example, to unicode
        if cur_array.dtype.type != ref_array.dtype.type:
            cur_array = cur_array.astype('U')
            try:
                ref_array = ref_array.astype('U')
            except:
                # ref_array of object type and each element must be utf-8 decoded
                utf8_decoded_elems = [elem.decode('UTF-8') for elem in ref_array.flatten()]
                ref_array = np.array(utf8_decoded_elems, dtype=str).reshape(ref_array.shape)
        return np.array_equal(cur_array, ref_array)
    elif cur_array.dtype == bool:
        abs_diff = np.absolute(cur_array ^ ref_array)
    else:
        abs_diff = np.absolute(cur_array - ref_array)
    max_val = np.maximum(np.absolute(cur_array), np.absolute(ref_array))
    return ((abs_diff < atol) | (abs_diff < rtol * max_val)).all()


def copy_files_by_pattern(directory: Path, pattern_to_find: str, pattern_to_copy: str):
    for file in directory.glob("{}*".format(pattern_to_find)):
        file_extension = ''.join(file.suffixes)
        copied_file = file.parent / (pattern_to_copy + file_extension)
        if not copied_file.exists() and file.exists():
            logging.info('Copying file from {} to {}'.format(file, copied_file))
            shutil.copy(str(file), str(copied_file))
        else:
            logging.info('File {} already exist or file {} does not exist'.format(copied_file, file))
