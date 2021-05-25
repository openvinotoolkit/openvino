import logging
import os
import subprocess
import sys

import cv2
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


def preprocess_image(image_path, img_size=None, scale=1, mean=0, flip=False):
    if mean == ():
        mean = 0
    if flip:
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(image_path)

    if img_size is not None and len(img_size) > 1:
        img = cv2.resize(img, img_size)
    else:
        if img_size != np.prod(img.shape):
            raise RuntimeError('Image size {} of image: {} not equal with model input shape {}. '
                               'Please use dedicated image for this topology'
                               .format(np.prod(img.shape), image_path, img_size[0]))
    img = img.astype(np.float32)
    img = img - mean
    img = img / scale
    return img


def preprocess_input(input_paths, scale=1, mean=0, flip=False, inputs=None, framework=None):
    """
    Preprocess input file. Support reading mif files, images or npy files
    :param input_path: path to input file
    :return: dict of pairs input layer - data for mif. numpy array for npy or image input
    """
    net_input = {}

    # TODO: HOTFIX! Rewrite for use multi_image
    if type(input_paths) is not list:
        input_paths = [input_paths]
    for model_input in inputs:
        for input_path in input_paths:
            feed_dict = dict()
            if os.path.splitext(input_path)[1] in ('.mif', '.txt'):
                # TODO: add multiimage batch for mif
                with open(input_path, 'r') as f:
                    file = f.readlines()
                i = 0
                while i < len(file):
                    header = file[i].replace('\n', '')
                    input_info = header.split(' ')
                    input_name = input_info[0]
                    input_shape = tuple(
                        map(lambda x: int(x), input_info[1].replace('(', '').replace(')', '').split(',')))
                    input_form = input_info[-1]
                    if input_form == 'binary':
                        i += 1
                        data = np.array(file[i].split(' '), dtype=np.float32)
                        data = np.reshape(data, input_shape)
                    elif os.path.isfile(os.path.join(os.path.dirname(input_path), input_form)) and \
                            os.path.splitext(input_form)[1] in ('.jpg', '.bmp', '.png'):

                        # For each input its own mean_values (that was set in test class)
                        if type(mean) is dict:
                            mean_values = mean[input_name]
                        else:
                            mean_values = mean
                        # Size (24,24) has len more than 1. Case with len=1: if instead of (24,24) we use 24*24*3*1
                        if len(input_shape) > 1:
                            data = preprocess_image(image_path=os.path.join(os.path.dirname(input_path), input_form),
                                                    img_size=(input_shape[1], input_shape[0]), scale=scale,
                                                    mean=mean_values,
                                                    flip=flip)
                        else:
                            data = preprocess_image(image_path=os.path.join(os.path.dirname(input_path), input_form),
                                                    img_size=input_shape, scale=scale,
                                                    mean=mean_values,
                                                    flip=flip)
                        data = np.expand_dims(data, axis=0)
                    elif os.path.splitext(input_form)[1] in ('.npy'):
                        input_form = os.path.join(os.path.dirname(input_path), input_form)
                        data = np.load(input_form)
                    else:
                        raise RuntimeError("No binary, no image and no npy format here: {}".format(input_form))
                    feed_dict[input_name] = data
                    i += 1

            elif os.path.isfile(input_path) and os.path.splitext(input_path)[1] in ('.jpg', '.bmp', '.png'):
                # if len(inputs) > 1:
                #    raise RuntimeError("Can't use image for multi input topologie!\nPlease use multi input file format")
                # else:
                if framework is not None:
                    shape = list(inputs[model_input]['shape'])
                    _, _, height, width = nchw_from_model(framework=framework, input_shape=shape)
                else:
                    shape = list(inputs[model_input]['shape'])
                    width = max(shape)
                    shape.remove(width)
                    height = max(shape)
                data = preprocess_image(image_path=input_path, img_size=(width, height),
                                        scale=scale, mean=mean, flip=flip)
                data = np.expand_dims(data, axis=0)
                feed_dict[model_input] = data
            elif os.path.isfile(input_path) and os.path.splitext(input_path)[1] in ('.npy'):
                data = np.load(input_path)
                feed_dict[model_input] = data
            elif os.path.isfile(input_path) and os.path.splitext(input_path)[1] in ('.npz'):
                data = np.load(input_path)
                for _input in data.files:
                    feed_dict[_input] = data[_input]
            if model_input not in net_input:
                net_input[model_input] = feed_dict[model_input]
            else:
                net_input[model_input] = np.append(net_input[model_input], feed_dict[model_input], axis=0)
    return net_input


def softmax(val, axis=-1):
    """ compute the softmax of the given tensor, normalizing on axis """
    exp = np.exp(val - np.amax(val, axis=axis, keepdims=True))
    return exp / np.sum(exp, axis=axis, keepdims=True)


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2.
    l2 = x2 - w2 / 2.
    left = max(l1, l2)
    r1 = x1 + w1 / 2.
    r2 = x2 + w2 / 2.
    right = min(r1, r2)
    return right - left


def box_intersection(ax, ay, aw, ah, bx, by, bw, bh):
    w = overlap(ax, aw, bx, bw)
    h = overlap(ay, ah, by, bh)
    if w < 0 or h < 0: return 0
    area = w * h
    return area


def box_union(ax, ay, aw, ah, bx, by, bw, bh):
    i = box_intersection(ax, ay, aw, ah, bx, by, bw, bh)
    u = aw * ah + bw * bh - i
    return u


def nchw_from_model(framework: str, input_shape):
    """
    Function to return values from framework input shape in (n, c, h, w) format
    :param framework: name of the framework
    :param input_shape: shape of the model input
    :return: batch, channel, height, width in (n, c, h, w) format
    """
    height_width_layout = {'caffe': (0, 1, 2, 3), 'tf': (0, 3, 1, 2), 'mxnet': (0, 1, 2, 3)}
    batch_index = height_width_layout[framework][0]
    channel_index = height_width_layout[framework][1]
    height_index = height_width_layout[framework][2]
    width_index = height_width_layout[framework][3]
    batch = input_shape[batch_index]
    channel = input_shape[channel_index]
    height = input_shape[height_index]
    width = input_shape[width_index]

    return batch, channel, height, width


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
