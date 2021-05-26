import logging
import os
import subprocess
import sys

import numpy as np


logger = logging.getLogger(__name__)


def generate_ir(coverage=False, **kwargs):
    if os.environ.get("MO_RUN_MODE") and os.environ.get("MO_RUN_MODE").lower() != "cmd":
        raise EnvironmentError("Unexpected value of MO_RUN_MODE variable. We support only 'cmd' testing of MO")
    print("Model Optimizer will be run via command line<br>\n")

    # Get default mo args
    mo = os.path.join(os.environ.get("MO_ROOT"), "mo.py")
    if coverage:
        params = [sys.executable, '-m', 'coverage', 'run', '-p', '--source={}'.format(os.environ.get("MO_ROOT")),
                  '--omit=*_test.py', mo]
    else:
        params = [sys.executable, mo]
    for key, value in kwargs.items():
        if key == "batch":
            params.extend(("-b", str(value)))
        elif key == "k":
            params.extend(("-k", str(value)))
        elif isinstance(value, bool) and value:
            params.append("--{}".format(key))
        elif isinstance(value, bool) and not value:
            continue
        elif (key == "input_model" or key == "input_shape" or key == "mean_file_offsets") \
                and ((isinstance(value, tuple) and value != ())
                     or (isinstance(value, str))):
            params.extend(("--{}".format(key), str('"{}"'.format(value))))
        elif (key == "mean_values" and (' ' in value or '(' in value)):
            params.extend(("--{}".format(key), str('"{}"'.format(value))))
        else:
            params.extend(("--{}".format(key), str(value)))
    exit_code, stdout, stderr = shell(params)
    logger.info("Model Optimizer out:\n{}".format(stdout))
    logger.error(stderr)
    return exit_code, stderr


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


def softmax(val, axis=-1):
    """ compute the softmax of the given tensor, normalizing on axis """
    exp = np.exp(val - np.amax(val, axis=axis, keepdims=True))
    return exp / np.sum(exp, axis=axis, keepdims=True)


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
    abs_diff = np.absolute(cur_array - ref_array)
    max_val = np.maximum(np.absolute(cur_array), np.absolute(ref_array))
    return ((abs_diff < atol) | (abs_diff < rtol * max_val)).all()
