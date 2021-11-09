# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import cv2
import re
import numpy as np
from glob import glob
from collections import defaultdict
from pathlib import Path
from itertools import chain

from .constants import IMAGE_EXTENSIONS, BINARY_EXTENSIONS
from .logging import logger


def set_inputs(paths_to_input, batch_size, app_input_info, requests):
  requests_input_data = get_inputs(paths_to_input, batch_size, app_input_info, requests)
  for i in range(len(requests)):
    inputs = requests[i].input_blobs
    for k, v in requests_input_data[i].items():
        if k not in inputs.keys():
            raise Exception(f"No input with name {k} found!")
        inputs[k].buffer[:] = v


def get_inputs(paths_to_input, batch_size, app_input_info, requests):
    input_file_mapping = parse_paths_to_input(paths_to_input)
    check_input_file_mapping(input_file_mapping, app_input_info)

    input_image_sizes = {}
    for key in sorted(app_input_info.keys()):
        info = app_input_info[key]
        if info.is_image:
            input_image_sizes[key] = (info.width, info.height)

    images_count = len(input_image_sizes.keys())
    binaries_count = len(app_input_info) - images_count

    image_files = list()
    binary_files = list()

    if paths_to_input and not input_file_mapping:
        image_files = get_files_by_extensions(paths_to_input, IMAGE_EXTENSIONS)
        binary_files = get_files_by_extensions(paths_to_input, BINARY_EXTENSIONS)

    if input_file_mapping and len(input_file_mapping) < len(app_input_info):
        not_provided_inputs = set(app_input_info) - set(input_file_mapping)
        logger.warning("No input files were given for the inputs: "
                       f"{', '.join(not_provided_inputs)}. This inputs will be filled with random values!")
    elif (len(image_files) == 0) and (len(binary_files) == 0):
        logger.warning("No input files were given: all inputs will be filled with random values!")
    else:
        binary_to_be_used = binaries_count * batch_size * len(requests)
        if binary_to_be_used > 0 and len(binary_files) == 0:
            logger.warning(f"No supported binary inputs found! "
                                        f"Please check your file extensions: {','.join(BINARY_EXTENSIONS)}")
        elif binary_to_be_used > len(binary_files):
            logger.warning(
                f"Some binary input files will be duplicated: "
                                        f"{binary_to_be_used} files are required, "
                                        f"but only {len(binary_files)} were provided")
        elif binary_to_be_used < len(binary_files):
            logger.warning(
                f"Some binary input files will be ignored: only {binary_to_be_used} "
                                        f"files are required from {len(binary_files)}")

        images_to_be_used = images_count * batch_size * len(requests)
        if images_to_be_used > 0 and len(image_files) == 0:
            logger.warning(f"No supported image inputs found! Please check your "
                                        f"file extensions: {','.join(IMAGE_EXTENSIONS)}")
        elif images_to_be_used > len(image_files):
            logger.warning(
                f"Some image input files will be duplicated: {images_to_be_used} "
                            f"files are required, but only {len(image_files)} were provided")
        elif images_to_be_used < len(image_files):
            logger.warning(
                f"Some image input files will be ignored: only {images_to_be_used} "
                                                    f"files are required from {len(image_files)}")

    requests_input_data = []
    for request_id in range(0, len(requests)):
        logger.info(f"Infer Request {request_id} filling")
        input_data = {}
        keys = list(sorted(app_input_info.keys()))
        for key in keys:
            info = app_input_info[key]
            if info.is_image:
                # input is image
                if key in input_file_mapping:
                    input_data[key] = fill_blob_with_image(input_file_mapping[key], request_id, batch_size,
                                                           keys.index(key), len(keys), info, from_map=True)
                    continue

                if len(image_files) > 0:
                    input_data[key] = fill_blob_with_image(image_files, request_id, batch_size, keys.index(key),
                                                           len(keys), info)
                    continue

            # input is binary
            if len(binary_files) or key in input_file_mapping:
                if key in input_file_mapping:
                    input_data[key] = fill_blob_with_binary(input_file_mapping[key], request_id, batch_size,
                                                           keys.index(key), len(keys), info, from_map=True)
                    continue

                input_data[key] = fill_blob_with_binary(binary_files, request_id, batch_size, keys.index(key),
                                                        len(keys), info)
                continue

            # most likely input is image info
            if info.is_image_info and len(input_image_sizes) == 1:
                image_size = input_image_sizes[list(input_image_sizes.keys()).pop()]
                logger.info("Fill input '" + key + "' with image size " + str(image_size[0]) + "x" +
                            str(image_size[1]))
                input_data[key] = fill_blob_with_image_info(image_size, info)
                continue

            # fill with random data
            logger.info(f"Fill input '{key}' with random values "
                                    f"({'image' if info.is_image else 'some binary data'} is expected)")
            input_data[key] = fill_blob_with_random(info)

        requests_input_data.append(input_data)

    return requests_input_data


def get_files_by_extensions(paths_to_input, extensions):
    if len(paths_to_input) == 1:
        files = [file for file in paths_to_input[0].split(",") if file]

        if all(get_extension(file) in extensions for file in files):
            check_files_exist(files)
            return files

    return get_files_by_extensions_for_not_list_of_files(paths_to_input, extensions)


def get_files_by_extensions_for_not_list_of_files(paths_to_input, extensions):
    input_files = list()

    for path_to_input in paths_to_input:
        if os.path.isfile(path_to_input):
            files = [os.path.normpath(path_to_input)]
        else:
            path = os.path.join(path_to_input, '*')
            files = glob(path, recursive=True)
        for file in files:
            file_extension = get_extension(file)
            if file_extension in extensions:
                input_files.append(file)
        input_files.sort()
        return input_files


def get_extension(file_path):
    return file_path.split(".")[-1].upper()


def fill_blob_with_image(image_paths, request_id, batch_size, input_id, input_size, info, from_map=False):
    shape = info.shape
    images = np.ndarray(shape)
    if from_map:
        image_index = request_id * batch_size
    else:
        image_index = request_id * batch_size * input_size + input_id

    scale_mean = (not np.array_equal(info.scale, (1.0, 1.0, 1.0)) or not np.array_equal(info.mean, (0.0, 0.0, 0.0)))

    for b in range(batch_size):
        image_index %= len(image_paths)
        image_filename = image_paths[image_index]
        logger.info(f'Prepare image {image_filename}')
        image = cv2.imread(image_filename)
        new_im_size = tuple((info.width, info.height))
        if image.shape[:-1] != new_im_size:
            logger.warning(f"Image is resized from ({image.shape[:-1]}) to ({new_im_size})")
            image = cv2.resize(image, new_im_size)

        if scale_mean:
            blue, green, red = cv2.split(image)
            blue = np.subtract(blue, info.mean[0])
            blue = np.divide(blue, info.scale[0])
            green = np.subtract(green, info.mean[1])
            green = np.divide(green, info.scale[1])
            red = np.subtract(red, info.mean[2])
            red = np.divide(red, info.scale[2])
            image = cv2.merge([blue, green, red])

        if info.layout in ['NCHW', 'CHW']:
            image = image.transpose((2, 0, 1))

        images[b] = image

        if from_map:
            image_index += 1
        else:
            image_index += input_size
    return images

def get_dtype(precision):
    format_map = {
      'FP32' : (np.float32, np.finfo(np.float32).min, np.finfo(np.float32).max),
      'I32'  : (np.int32, np.iinfo(np.int32).min, np.iinfo(np.int32).max),
      'I64'  : (np.int64, np.iinfo(np.int64).min, np.iinfo(np.int64).max),
      'FP16' : (np.float16, np.finfo(np.float16).min, np.finfo(np.float16).max),
      'I16'  : (np.int16, np.iinfo(np.int16).min, np.iinfo(np.int16).max),
      'U16'  : (np.uint16, np.iinfo(np.uint16).min, np.iinfo(np.uint16).max),
      'I8'   : (np.int8, np.iinfo(np.int8).min, np.iinfo(np.int8).max),
      'U8'   : (np.uint8, np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
      'BOOL' : (np.uint8, 0, 1),
    }
    if precision in format_map.keys():
        return format_map[precision]
    raise Exception("Can't find data type for precision: " + precision)

def fill_blob_with_binary(binary_paths, request_id, batch_size, input_id, input_size, info, from_map=False):
    binaries = np.ndarray(info.shape)
    shape = info.shape.copy()
    if 'N' in info.layout:
        shape[info.layout.index('N')] = 1
    if from_map:
        binary_index = request_id * batch_size
    else:
        binary_index = request_id * batch_size * input_size + input_id
    dtype = get_dtype(info.precision)[0]
    for b in range(batch_size):
        binary_index %= len(binary_paths)
        binary_filename = binary_paths[binary_index]
        logger.info("Prepare binary file " + binary_filename)

        binary_file_size = os.path.getsize(binary_filename)
        blob_size = dtype().nbytes * int(np.prod(shape))
        if blob_size != binary_file_size:
            raise Exception(
                f"File {binary_filename} contains {binary_file_size} bytes but network expects {blob_size}")
        binaries[b] = np.reshape(np.fromfile(binary_filename, dtype), shape)

        if from_map:
            binary_index += 1
        else:
            binary_index += input_size

    return binaries


def fill_blob_with_image_info(image_size, layer):
    shape = layer.shape
    im_info = np.ndarray(shape)
    for b in range(shape[0]):
        for i in range(shape[1]):
            im_info[b][i] = image_size[i] if i in [0, 1] else 1

    return im_info

def fill_blob_with_random(layer):
    dtype, rand_min, rand_max = get_dtype(layer.precision)
    # np.random.uniform excludes high: add 1 to have it generated
    if np.dtype(dtype).kind in ['i', 'u', 'b']:
        rand_max += 1
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    if layer.shape:
        return rs.uniform(rand_min, rand_max, layer.shape).astype(dtype)
    return (dtype)(rs.uniform(rand_min, rand_max))


def parse_paths_to_input(paths_to_inputs):
    input_dicts_list = [parse_path(path) for path in paths_to_inputs]
    inputs = defaultdict(list)
    for input_dict in input_dicts_list:
        for input_name, input_files in input_dict.items():
            inputs[input_name] += input_files
    return {input_: files for input_, files in inputs.items() if files}


def parse_path(path):
    """
    Parse "input_1:file1,file2,input_2:file3" into a dict
    """
    inputs = re.findall(r"([^,]\w+):", path)
    input_files = [file for file in re.split(r"[^,]\w+:", path) if file]
    return {
        input_: files.strip(",").split(",") for input_, files in zip(inputs, input_files)
    }


def check_input_file_mapping(input_file_mapping, app_input_info):
    check_inputs(app_input_info, input_file_mapping)
    check_input_file_mapping_files_exists(input_file_mapping)
    check_files_extensions(app_input_info, input_file_mapping)


def check_inputs(app_input_info, input_file_mapping):
    wrong_inputs = [
        input_ for input_ in input_file_mapping if input_ not in app_input_info
    ]
    if wrong_inputs:
        raise Exception(
            f"Wrong input mapping! Cannot find inputs: {wrong_inputs}. "
            f"Available inputs: {list(app_input_info)}. "
            "Please check `-i` input data"
        )


def check_input_file_mapping_files_exists(input_file_mapping):
    check_files_exist(chain.from_iterable(input_file_mapping.values()))


def check_files_exist(input_files_list):
    not_files = [
        file for file in input_files_list if not Path(file).is_file()
    ]
    if not_files:
        not_files = ",\n".join(not_files)
        raise Exception(
            f"Inputs are not files or does not exist!\n {not_files}"
        )


def check_files_extensions(app_input_info, input_file_mapping):
    unsupported_files = []
    for input_, files in input_file_mapping.items():
        info = app_input_info[input_]

        proper_extentions = IMAGE_EXTENSIONS if info.is_image else BINARY_EXTENSIONS
        unsupported = "\n".join(
                [file for file in files if Path(file).suffix.upper().strip(".") not in proper_extentions]
            )
        if unsupported:
            unsupported_files.append(unsupported)
    if unsupported_files:
        unsupported_files = "\n".join(unsupported_files)
        raise Exception(
            f"This files has unsupported extensions: {unsupported_files}.\n"
            f"Supported extentions:\nImages: {IMAGE_EXTENSIONS}\nBinary: {BINARY_EXTENSIONS}"
        )
