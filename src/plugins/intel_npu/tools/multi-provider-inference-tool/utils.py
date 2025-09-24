#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import math
import os

import cv2
import numpy as np

def prepare_input_description(infiles, data_shape, model_data_type, layout):
    infiles_description = copy.deepcopy(infiles)
    if "layout" not in infiles_description.keys():
        infiles_description["layout"] = layout

    if infiles_description["type"] == "image":
        if "convert" in infiles_description.keys():
            if "shape" in infiles_description["convert"].keys():
                infiles_description["to_shape"] = infiles_description["convert"]["shape"]
            if "element_type" in infiles_description["convert"].keys():
                infiles_description["to_model_element_type"] = infiles_description["convert"]["element_type"]

    if "to_shape" not in infiles_description.keys():
        infiles_description["to_shape"] = data_shape

    # both image & bin must know a model element type to expect
    if "to_model_element_type" not in infiles_description.keys():
        infiles_description["to_model_element_type"] = model_data_type
    return infiles_description


def load_tensor_from_file(infiles_description):
    if "element_type" not in infiles_description.keys():
        raise RuntimeError('In case of binary input each input description must be accompanied by "element_type". Please add it to each input data')
    file_data_precision = np.dtype(infiles_description["element_type"])
    tensor_shape = infiles_description["shape"]
    layout = infiles_description["layout"]
    model_expected_precision = np.dtype(infiles_description["to_model_element_type"])
    files_list = infiles_description["files"]

    batch_dimension_index_in_tensor_shape_or_none = None
    try:
        batch_dimension_index_in_tensor_shape_or_none = layout.lower().index("n")
    except ValueError:
        pass

    if len(files_list) == 0:
        return None, infiles_description
    assert len(files_list) == 1, "Batched loading is unsupported"

    # TODO iterate over files list to process different lines of batch in case of batched input (fixme)
    file_path = files_list.pop(0)
    infiles_description["files"] = files_list

    # In case of binary input, we must ensure that binary files size fit to tensor required size
    file_size = os.path.getsize(file_path)
    requested_tensor_size = math.prod(tensor_shape) * model_expected_precision.itemsize
    file_tensor_size = math.prod(tensor_shape) * file_data_precision.itemsize

    if model_expected_precision != file_data_precision:
        print(f"Converting {file_path} input from {file_data_precision} to {model_expected_precision}")
        # binary file fit to tensor
        if file_size == file_tensor_size:
            tensor_from_file = np.fromfile(file_path, dtype=file_data_precision).reshape(tensor_shape)
            assert model_expected_precision == file_data_precision, "TODO convert tensors"
            requested_tensor = tensor_from_file
            return requested_tensor, infiles_description

        # When their sizes are not equal, check if binary file can be a part of a batched tensor
        print(
            f"File contains {file_size} bytes, but it expected to be: {file_tensor_size} while converting precision from {file_data_precision.name} to {model_expected_precision.name}. Check whether it is possible to fit it into batch loading "
        )
        assert batch_dimension_index_in_tensor_shape_or_none is not None, f"Input layout has no batch dimension: {layout}"
        N = tensor_shape[batch_dimension_index_in_tensor_shape_or_none]
        assert (
            file_size * N == file_tensor_size
        ), f"File {file_path} contains {file_size} bytes, but {file_tensor_size} total in batch size {N} expected while converting precision from {file_data_precision.name} to {model_expected_precision.name}"

        debatched_shape_list = copy.deepcopy(tensor_shape)
        debatched_shape_list[batch_dimension_index_in_tensor_shape_or_none] = 1
        tensor_from_file = np.fromfile(file_path, dtype=file_data_precision).reshape(debatched_shape_list)
        assert model_expected_precision == file_data_precision, "TODO convert tensors for batch"
        assert file_index_in_batch, "Batch is unsupported at the moment"
        requested_tensor = tensor_from_file
        return requested_tensor, infiles_description
    else:
        # binary file fit to tensor
        if file_size == requested_tensor_size:
            return np.fromfile(file_path, dtype=file_data_precision).reshape(tensor_shape), infiles_description

        # When their sizes are not equal, check if binary file can be a part of a batched tensor
        print(
            f"File contains {file_size} bytes, but it expected to be: {requested_tensor_size} when datatypes match. Check whether it is possible to fit it into batch loading"
        )
        assert batch_dimension_index_in_tensor_shape_or_none, f"Input layout has no batch dimension: {layout}"
        N = tensor_shape[batch_dimension_index_in_tensor_shape_or_none]
        assert (
            file_size * N == requested_tensor_size
        ), f"File {file_path} contains {file_size} bytes, but {requested_tensor_size} total in batch size {batch_dimension_index_in_tensor_shape_or_none} expected"

        debatched_shape_list = copy.deepcopy(tensor_shape)
        debatched_shape_list[batch_dimension_index_in_tensor_shape_or_none] = 1
        tensor_from_file = np.fromfile(file_path, dtype=file_data_precision).reshape(debatched_shape_list)
        assert file_index_in_batch, "Batch is unsupported at the moment"
        requested_tensor = tensor_from_file
        return requested_tensor, infiles_description


def load_image_from_file(infiles_description):
    files_list = infiles_description["files"]
    layout = infiles_description["layout"].upper()
    tensor_shape = infiles_description["to_shape"]
    model_expected_precision = np.dtype(infiles_description["to_model_element_type"])

    compatible_layouts = ["NHWC", "NCHW"]
    if layout not in compatible_layouts:
        raise RuntimeError(f"Incorrect layout: {layout}, expected: {compatible_layouts}")

    if len(files_list) == 0:
        return None, infiles_description
    assert len(files_list) == 1, "Batched loading is unsupported"

    # TODO iterate over files list to process different lines of batch in case of batched input (fixme)
    file_path = files_list.pop(0)
    infiles_description["files"] = files_list

    image = cv2.imread(file_path)

    # the image have to fit the model input shape
    h = tensor_shape[layout.index("H")]
    w = tensor_shape[layout.index("W")]
    resized_image = None
    try:
        if int(h) == -1 or int(w) == -1:
            # do not need to resize image as probably we deal with a dynamic shape
            resized_image = image
    except Exception:
        # do not need to resize image as probably we deal with a dynamic shape
        resized_image = image
        pass
    if resized_image is None:
        resized_image = cv2.resize(image, (w, h))

    # transpose image to a valid model layout
    if layout == "NCHW":
        transposed_image = np.transpose(resized_image, (2, 0, 1))
    else:
        transposed_image = np.asarray(resized_image, np.ubyte).astype(model_expected_precision)

    # convert image into proper datatype
    transposed_image = transposed_image.astype(model_expected_precision)

    # Add N dimension
    input_tensor = np.expand_dims(transposed_image, 0)
    return input_tensor, infiles_description


def load_objects_from_file(infiles_description):
    tensor_raw_array = None
    if infiles_description["type"] == "bin":
        tensor_raw_array, infiles_description = load_tensor_from_file(infiles_description)
    elif infiles_description["type"] == "image":
        tensor_raw_array, infiles_description = load_image_from_file(infiles_description)
    else:
        raise RuntimeError(f"Incorrect \"type\": {infiles_description['type']} is used as a source. Expected types: \"bin\",\"image\"")
    return tensor_raw_array, infiles_description


def get_model_name(model_path):
    return os.path.basename(model_path).split(".")[0]


def get_layout_from_shape(shape: list) -> str:
    rank = len(shape)
    if rank == 0:
        return ""
    if rank == 1:
        return "C"
    if rank == 2:
        return "NC"
    if rank == 3:
        return "CHW"
    if rank == 4:
        return "NCHW"
    if rank == 5:
        return "NCDHW"

    raise RuntimeError(f"Failed to get layout for shape: {shape.to_string()}")
