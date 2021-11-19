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
from numpy.core.numeric import allclose

from openvino import Tensor

from .constants import IMAGE_EXTENSIONS, BINARY_EXTENSIONS
from .logging import logger


class DataQueue:
    def __init__(self, input_data: dict, batch_sizes: list):
        self.input_data = input_data
        self.current_batch_id = 0
        self.size = len(batch_sizes)
        self.batch_sizes = batch_sizes

    def get_next_input(self):
        data = {}
        for input_name, input_tensors in self.input_data.items():
            data[input_name] = input_tensors[self.current_batch_id]
        self.current_batch_id = (self.current_batch_id+ 1) % self.size
        return data

    def get_next_batch_size(self):
        return self.batch_sizes[self.current_batch_id]


def get_batch_sizes(app_input_info):
    batch_sizes = []
    num_shapes = len(app_input_info[0].tensor_shapes)
    for i in range(num_shapes):
        batch_size = 0
        for info in app_input_info:
            batch_index = info.layout.index('N') if 'N' in info.layout else -1
            if batch_index != -1:
                shape = info.tensor_shapes[i]
                if batch_size == 0:
                    batch_size = shape[batch_index]
                elif batch_size != shape[batch_index]:
                    raise Exception("Can't deterimine batch size: batch is different for different inputs!")
        if batch_size == 0:
            batch_size = 1
        batch_sizes.append(batch_size)
    return batch_sizes


def get_input_data(paths_to_input, app_input_info):
    input_file_mapping = parse_paths_to_input(paths_to_input)
    check_input_file_mapping(input_file_mapping, app_input_info)

    image_sizes = get_image_sizes(app_input_info)
    batch_sizes = get_batch_sizes(app_input_info)
    images_count = 0
    binaries_count = 0
    image_sizes = {}
    for info in app_input_info:
        if info.is_image:
            image_sizes[info.name] = []
            for w, d in zip(info.widthes, info.heights):
                image_sizes[info.name].append((w, d))
                images_count += 1
        elif len(image_sizes) != 1 or not info.is_image_info:
            binaries_count += len(info.tensor_shapes)

    image_files = list()
    binary_files = list()

    if paths_to_input and not input_file_mapping:
        image_files = get_files_by_extensions(paths_to_input, IMAGE_EXTENSIONS)
        binary_files = get_files_by_extensions(paths_to_input, BINARY_EXTENSIONS)

    total_frames = np.prod(batch_sizes)

    if input_file_mapping and len(input_file_mapping) < len(app_input_info):
        not_provided_inputs = set(app_input_info) - set(input_file_mapping)
        logger.warning("No input files were given for the inputs: "
                       f"{', '.join(not_provided_inputs)}. This inputs will be filled with random values!")
    elif (len(image_files) == 0) and (len(binary_files) == 0):
        logger.warning("No input files were given: all inputs will be filled with random values!")
    else:
        max_binary_can_be_used = binaries_count * total_frames
        if max_binary_can_be_used > 0 and len(binary_files) == 0:
            logger.warning(f"No supported binary inputs found! "
                                        f"Please check your file extensions: {','.join(BINARY_EXTENSIONS)}")
        if max_binary_can_be_used > len(binary_files):
            logger.warning(
                f"Some binary input files will be duplicated: "
                                        f"{max_binary_can_be_used} files are required, "
                                        f"but only {len(binary_files)} were provided")

        max_images_can_be_used = images_count * total_frames
        if max_images_can_be_used > 0 and len(image_files) == 0:
            logger.warning(f"No supported image inputs found! Please check your "
                                        f"file extensions: {','.join(IMAGE_EXTENSIONS)}")
        elif max_images_can_be_used > len(image_files):
            logger.warning(
                f"Some image input files will be duplicated: {max_images_can_be_used} "
                            f"files are required, but only {len(image_files)} were provided")

    data = {}
    num_inputs = len(app_input_info)
    for input_id, info in enumerate(app_input_info):
        if info.is_image:
            # input is image
            if info.name in input_file_mapping:
                data[info.name] = fill_blob_with_image(input_file_mapping[info.name], info, batch_sizes, input_id, num_inputs, from_map=True)
                continue

            if len(image_files) > 0:
                data[info.name] = fill_blob_with_image(image_files, info, batch_sizes, input_id, num_inputs)
                continue

        if len(binary_files) or info.name in input_file_mapping:
            if info.name in input_file_mapping:
                data[info.name] = fill_blob_with_binary(input_file_mapping[info.name], info, batch_sizes, input_id, num_inputs, from_map=True)
                continue

            data[info.name] = fill_blob_with_binary(binary_files, info, batch_sizes, input_id, num_inputs)
            continue

        if info.is_image_info and len(image_sizes) == 1:
                image_size = image_sizes[0]
                logger.info(f"Create input tensors for input '{info.name}' with image sizes: {image_size}")
                data[info.name] = fill_blob_with_image_info(image_size, info)
                continue

        # fill with random data
        logger.info(f"Fill input '{info.name}' with random values "
                                f"({'image' if info.is_image else 'some binary data'} is expected)")
        data[info.name] = fill_blob_with_random(info)

    return DataQueue(data, batch_sizes)


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


def fill_blob_with_image(image_paths, info, batch_sizes, input_id, num_inputs, from_map=False):
    processed_frames = 0
    widthes, heights = info.widthes, info.heights
    tensors = []
    for i in range(len(info.tensor_shapes)):
        shape = list(info.tensor_shapes[i])
        images = np.ndarray(shape=shape, dtype=np.uint8)
        if from_map:
            image_index = processed_frames
        else:
            image_index = processed_frames * num_inputs + input_id

        scale_mean = (not np.array_equal(info.scale, (1.0, 1.0, 1.0)) or not np.array_equal(info.mean, (0.0, 0.0, 0.0)))

        current_batch_size = batch_sizes[i]
        for b in range(current_batch_size):
            image_index %= len(image_paths)
            image_filename = image_paths[image_index]
            logger.info(f'Prepare image {image_filename}')
            image = cv2.imread(image_filename)
            new_im_size = tuple((widthes[i], heights[i]))
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
                image_index += num_inputs
        processed_frames += current_batch_size
        tensors.append(Tensor(images))
    return tensors


def get_dtype(precision):
    format_map = {
      'f32' : (np.float32, np.finfo(np.float32).min, np.finfo(np.float32).max),
      'i32'  : (np.int32, np.iinfo(np.int32).min, np.iinfo(np.int32).max),
      'i64'  : (np.int64, np.iinfo(np.int64).min, np.iinfo(np.int64).max),
      'fp16' : (np.float16, np.finfo(np.float16).min, np.finfo(np.float16).max),
      'i16'  : (np.int16, np.iinfo(np.int16).min, np.iinfo(np.int16).max),
      'u16'  : (np.uint16, np.iinfo(np.uint16).min, np.iinfo(np.uint16).max),
      'i8'   : (np.int8, np.iinfo(np.int8).min, np.iinfo(np.int8).max),
      'u8'   : (np.uint8, np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
      'boolean' : (np.uint8, 0, 1),
    }
    if precision in format_map.keys():
        return format_map[precision]
    raise Exception("Can't find data type for precision: " + precision)


def fill_blob_with_binary(binary_paths, info, batch_sizes, input_id, num_inputs, from_map=False):
    processed_frames = 0
    tensors = []
    for i in range(len(info.tensor_shapes)):
        dtype = get_dtype(info.element_type.get_type_name())[0]
        binaries = np.ndarray(shape=list(info.tensor_shapes[i]), dtype=dtype)
        shape = binaries.copy()
        if 'N' in info.layout:
            shape[info.layout.index('N')] = 1
        if from_map:
            binary_index = processed_frames
        else:
            binary_index = processed_frames * num_inputs + input_id
        current_batch_size = batch_sizes[i]
        for b in range(current_batch_size):
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
                binary_index += num_inputs
        processed_frames += current_batch_size
        tensors.append(Tensor(binaries))
    return tensors


def get_image_sizes(app_input_info):
    image_sizes = []
    for info in app_input_info:
        if info.is_image:
            info_image_sizes = []
            for w, h in zip(info.widthes, info.heights):
                info_image_sizes.append((w, h))
            image_sizes.append(info_image_sizes)
    return image_sizes


def fill_blob_with_image_info(image_sizes, layer):
    im_infos = []
    for shape, image_size in zip(layer.tensor_shapes, image_sizes):
        im_info = np.ndarray(shape)
        for b in range(shape[0]):
            for i in range(shape[1]):
                im_info[b][i] = image_size[i] if i in [0, 1] else 1
        im_infos.append(Tensor(im_info))
    return im_infos


def fill_blob_with_random(layer):
    dtype, rand_min, rand_max = get_dtype(layer.element_type.get_type_name())
    # np.random.uniform excludes high: add 1 to have it generated
    if np.dtype(dtype).kind in ['i', 'u', 'b']:
        rand_max += 1
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    input_tensors = []
    for shape in layer.tensor_shapes:
        if shape:
            input_tensors.append(Tensor(rs.uniform(rand_min, rand_max, list(shape)).astype(dtype)))
        else:
            input_tensors.append(Tensor(rs.uniform(rand_min, rand_max)))
    return input_tensors


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
    input_names = [info.name for info in app_input_info]
    wrong_inputs = [
        input_ for input_ in input_file_mapping if input_ not in input_names
    ]
    if wrong_inputs:
        raise Exception(
            f"Wrong input mapping! Cannot find inputs: {wrong_inputs}. "
            f"Available inputs: {list(input_names)}. "
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
        info = [info for info in app_input_info if info.name == input_][0]
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
