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
from openvino.impl import Shape

from .constants import IMAGE_EXTENSIONS, BINARY_EXTENSIONS
from .logging import logger


class DataQueue:
    def __init__(self, input_data: dict, batch_sizes: list):
        self.input_data = input_data
        self.sizes_map = {}
        for name, tensors in input_data.items():
            self.sizes_map[name] = len(tensors)
        self.index_map = defaultdict.fromkeys(input_data.keys(), 0)
        self.batch_sizes = batch_sizes
        self.size = len(batch_sizes)
        self.current_batch_id = 0

    def get_next_input(self):
        data = {}
        for input_name, input_tensors in self.input_data.items():
            data[input_name] = input_tensors[self.index_map[input_name]]
            self.index_map[input_name] = (self.index_map[input_name] + 1) % self.sizes_map[input_name]
        self.current_batch_id = (self.current_batch_id + 1) % self.size
        return data

    def get_next_batch_size(self):
        return self.batch_sizes[self.current_batch_id]


def get_batch_sizes(app_input_info):
    batch_sizes = []
    niter = max(len(info.shapes) for info in app_input_info)
    for i in range(niter):
        batch_size = 0
        for info in app_input_info:
            batch_index = info.layout.index('N') if 'N' in info.layout else -1
            if info.shapes:
                shape = info.shapes[i % len(info.shapes)]
                if batch_index != -1 and len(shape) == len(info.layout):
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
    image_info_count = 0
    for info in app_input_info:
        if info.is_image:
            images_count += 1
        elif info.is_image_info:
            image_info_count += 1
    if images_count == 1 and image_info_count:
        binaries_count = len(app_input_info) - images_count - image_info_count
    else:
         binaries_count = len(app_input_info) - images_count

    image_files = list()
    binary_files = list()

    if paths_to_input and not input_file_mapping:
        image_files = get_files_by_extensions(paths_to_input, IMAGE_EXTENSIONS)
        binary_files = get_files_by_extensions(paths_to_input, BINARY_EXTENSIONS)

    files_to_be_used = max(list(len(files) for files in input_file_mapping.values())) if input_file_mapping \
                                                                                      else len(image_files) + len(binary_files)

    if input_file_mapping:
        for info in app_input_info:
            num_shapes = len(info.shapes)
            if info.name in input_file_mapping and num_shapes != 0:
                num_files = len(input_file_mapping[info.name])
                if num_files > num_shapes and num_files % num_shapes != 0:
                    files_to_be_used = num_files - num_files % num_shapes
                    logger.warning(f"Number of provided files for input '{info.name}' is not a multiple of the number of"
                                f"provided tensor shapes. Only {files_to_be_used} files will be used for each input")
    else:
        if binaries_count + images_count > 1:
            raise Exception("Number of inputs more than one, provide input names for each file/folder")
        else:
            num_shapes = len(app_input_info[0].shapes)
            if num_shapes != 0:
                num_files = len(image_files) + len(binary_files)
                if num_files > num_shapes and num_files % num_shapes != 0:
                    files_to_be_used = num_files - num_files % num_shapes
                    logger.warning(f"Number of provided files for input '{app_input_info[0].name}' is not a multiple of the number of"
                                f"provided tensor shapes. Only {files_to_be_used} files will be used for each input")

    total_frames = np.prod(batch_sizes)

    if input_file_mapping and len(input_file_mapping) < len(app_input_info):
        not_provided_inputs = set(info.name for info in app_input_info) - set(input_file_mapping)
        logger.warning("No input files were given for the inputs: "
                       f"{', '.join(not_provided_inputs)}. This inputs will be filled with random values!")
    elif len(image_files) == 0 and len(binary_files) == 0:
        for info in app_input_info:
            if len(info.shapes) == 0:
                raise Exception("No input images were given, provide tensor_shape.")
        logger.warning("No input files were given: all inputs will be filled with random values!")
    else:
        max_binary_can_be_used = 0
        if total_frames: # if tensor shapes are defined already
            max_binary_can_be_used = binaries_count * total_frames
        if max_binary_can_be_used > 0 and len(binary_files) == 0:
            logger.warning(f"No supported binary inputs found! "
                                        f"Please check your file extensions: {','.join(BINARY_EXTENSIONS)}")
        elif max_binary_can_be_used > len(binary_files):
            logger.warning(
                f"Some binary input files will be duplicated: "
                                        f"{max_binary_can_be_used} files are required, "
                                        f"but only {len(binary_files)} were provided")

        max_images_can_be_used = 0
        if total_frames: # if tensor shapes are defined already
            max_images_can_be_used = images_count * total_frames
        if max_images_can_be_used > 0 and len(image_files) == 0:
            logger.warning(f"No supported image inputs found! Please check your "
                                        f"file extensions: {','.join(IMAGE_EXTENSIONS)}")
        elif max_images_can_be_used > len(image_files):
            logger.warning(
                f"Some image input files will be duplicated: {max_images_can_be_used} "
                            f"files are required, but only {len(image_files)} were provided")

    data = {}
    for info in app_input_info:
        if info.is_image:
            # input is image
            if info.name in input_file_mapping:
                data[info.name] = fill_blob_with_image(input_file_mapping[info.name][:files_to_be_used], info, batch_sizes)
                continue

            if len(image_files) > 0:
                data[info.name] = fill_blob_with_image(image_files[:files_to_be_used], info, batch_sizes)
                continue

        if len(binary_files) or info.name in input_file_mapping:
            if info.name in input_file_mapping:
                data[info.name] = fill_blob_with_binary(input_file_mapping[info.name][:files_to_be_used], info, batch_sizes)
                continue

            data[info.name] = fill_blob_with_binary(binary_files[:files_to_be_used], info, batch_sizes)
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
    if len(batch_sizes) == 0:
        batch_sizes = get_batch_sizes(app_input_info) # update batch sizes in case getting tensor shapes from images
    return DataQueue(data, batch_sizes)


def get_files_by_extensions(paths_to_input, extensions):
    if len(paths_to_input) == 1:
        files = [file for file in paths_to_input[0].split(",") if file]

        if all(get_extension(file) in extensions for file in files):
            check_files_exist(files)
            return files

    return get_files_by_extensions_for_directory_or_list_of_files(paths_to_input, extensions)


def get_files_by_extensions_for_directory_or_list_of_files(paths_to_input, extensions):
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


def fill_blob_with_image(image_paths, info, batch_sizes):
    processed_frames = 0
    widthes = info.widthes if info.is_dynamic else [len(info.width)]
    heights = info.heights if info.is_dynamic else [len(info.height)]
    tensors = []
    process_with_original_shapes = False
    num_shapes = len(info.shapes)
    if num_shapes == 0:
        process_with_original_shapes = True
    num_images = len(image_paths)
    niter = max(num_shapes, num_images)
    for i in range(niter):
        shape = list(info.shapes[i % num_shapes]) if num_shapes else []
        images = np.ndarray(shape=shape, dtype=np.uint8)
        image_index = processed_frames
        scale_mean = (not np.array_equal(info.scale, (1.0, 1.0, 1.0)) or not np.array_equal(info.mean, (0.0, 0.0, 0.0)))

        current_batch_size = 1 if process_with_original_shapes else batch_sizes[i % num_shapes]
        for b in range(current_batch_size):
            image_index %= num_images
            image_filename = image_paths[image_index]
            logger.info(f'Prepare image {image_filename}')
            image = cv2.imread(image_filename)
            if not process_with_original_shapes: # TODO: check if info.partial_shape is compatible with image shape
                new_im_size = (widthes[i % num_shapes], heights[i % num_shapes])
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

            if process_with_original_shapes:
                expanded = np.expand_dims(image, 0)
                info.tensor_shapes.append(Shape(expanded.shape))
                tensors.append(Tensor(expanded))
            else:
                images[b] = image
            image_index += 1
        processed_frames += current_batch_size
        if not process_with_original_shapes:
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


def fill_blob_with_binary(binary_paths, info, batch_sizes):
    num_shapes = len(info.shapes)
    if info.is_dynamic and num_shapes == 0:
        raise Exception("Tensor shapes must be specified for binary inputs and dynamic model")
    num_binaries = len(binary_paths)
    niter = max(num_shapes, num_binaries)
    processed_frames = 0
    tensors = []
    for i in range(niter):
        shape_id = i % num_shapes
        dtype = get_dtype(info.element_type.get_type_name())[0]
        shape = list(info.shapes[shape_id])
        binaries = np.ndarray(shape=shape, dtype=dtype)
        if 'N' in info.layout:
            shape[info.layout.index('N')] = 1
        binary_index = processed_frames
        current_batch_size = batch_sizes[shape_id]
        for b in range(current_batch_size):
            binary_index %= num_binaries
            binary_filename = binary_paths[binary_index]
            logger.info("Prepare binary file " + binary_filename)

            binary_file_size = os.path.getsize(binary_filename)
            blob_size = dtype().nbytes * int(np.prod(shape))
            if blob_size != binary_file_size:
                raise Exception(
                    f"File {binary_filename} contains {binary_file_size} bytes but network expects {blob_size}")
            binaries[b] = np.reshape(np.fromfile(binary_filename, dtype), shape)

            binary_index += 1
        processed_frames += current_batch_size
        tensors.append(Tensor(binaries))
    return tensors


def get_image_sizes(app_input_info):
    image_sizes = []
    for info in app_input_info:
        if info.is_image:
            if info.is_static:
                image_sizes.append([info.width, info.height])
            else:
                info_image_sizes = []
                for w, h in zip(info.widthes, info.heights):
                    info_image_sizes.append((w, h))
                image_sizes.append(info_image_sizes)
    return image_sizes


def fill_blob_with_image_info(image_sizes, layer):
    im_infos = []
    for shape, image_size in zip(layer.shapes, image_sizes):
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
    for shape in layer.shapes:
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
    check_number_of_parameters_for_each_input(input_file_mapping)


def check_number_of_parameters_for_each_input(input_parameters_mapping):
    num_parameters_for_each_input = list(len(input_parameters) for input_parameters in input_parameters_mapping.values() if len(input_parameters) != 1)
    if len(num_parameters_for_each_input) > 1:
        if num_parameters_for_each_input.count(num_parameters_for_each_input[0]) != len(num_parameters_for_each_input):
            raise Exception(
                "Files number for every input should be either 1 or should be equal to files number of other inputs"
            )


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
