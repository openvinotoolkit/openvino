#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import numpy as np


def calculate_nrmse(lhs_normalized_bin_array, rhs_normalized_bin_array):
    diff_img = lhs_normalized_bin_array - rhs_normalized_bin_array
    return float(
        1
        - np.sqrt(np.sum(diff_img * diff_img) / (len(lhs_normalized_bin_array.flatten())))
        / (
            max(
                0.001,
                max(0, (np.max(lhs_normalized_bin_array))) - min(0, np.min(lhs_normalized_bin_array)),
                max(0, (np.max(rhs_normalized_bin_array))) - min(0, np.min(rhs_normalized_bin_array)),
            )
        )
    )


def compare_typed_blobs(lhs_blob_file_path, rhs_blob_file_path, np_data_type):
    lhs_binary_array = np.fromfile(lhs_blob_file_path, np_data_type)
    rhs_binary_array = np.fromfile(rhs_blob_file_path, np_data_type)
    lhs_img_size = lhs_binary_array.nbytes
    rhs_img_size = rhs_binary_array.nbytes

    if lhs_img_size != rhs_img_size:
        raise RuntimeError(f"Blobs are not alligned, sizes: {lhs_img_size}/{rhs_img_size}, data type: {np_data_type}")

    return calculate_nrmse(lhs_binary_array.astype(np.float32), rhs_binary_array.astype(np.float32))


def normalize_float_array(array, np_type):
    array[np.isnan(array)] = 0
    array[np.isneginf(array)] = np.finfo(np_type).min
    array[np.isinf(array)] = np.finfo(np_type).max
    array[array < np.finfo(np_type).min] = np.finfo(np_type).min
    array[array > np.finfo(np_type).max] = np.finfo(np_type).max
    return array


def compare_fp16_blobs(lhs_blob_file_path, rhs_blob_file_path):
    lhs_binary_array = np.fromfile(lhs_blob_file_path, np.float16)
    lhs_binary_array = lhs_binary_array.astype(np.float32)
    rhs_binary_array = np.fromfile(rhs_blob_file_path, np.float16)
    rhs_binary_array = rhs_binary_array.astype(np.float32)

    lhs_img_size = lhs_binary_array.nbytes
    rhs_img_size = rhs_binary_array.nbytes

    if lhs_img_size != rhs_img_size:
        raise RuntimeError(f"Blobs are not alligned, sizes: {lhs_img_size}/{rhs_img_size}, data type: {np.float16}")

    rhs_binary_array = normalize_float_array(rhs_binary_array, np.float16)
    lhs_binary_array = normalize_float_array(lhs_binary_array, np.float16)
    return calculate_nrmse(lhs_binary_array, rhs_binary_array)


def compare_fp32_blobs(lhs_blob_file_path, rhs_blob_file_path):
    lhs_binary_array = np.fromfile(lhs_blob_file_path, np.float32)
    rhs_binary_array = np.fromfile(rhs_blob_file_path, np.float32)
    lhs_img_size = lhs_binary_array.nbytes
    rhs_img_size = rhs_binary_array.nbytes

    if lhs_img_size != rhs_img_size:
        raise RuntimeError(f"Blobs are not alligned, sizes: {lhs_img_size}/{rhs_img_size}, data type: {np.float32}")

    rhs_binary_array = normalize_float_array(rhs_binary_array, np.float32)
    lhs_binary_array = normalize_float_array(lhs_binary_array, np.float32)
    return calculate_nrmse(lhs_binary_array, rhs_binary_array)


def get_np_data_type(data_type: str):
    return np.dtype(getattr(np, data_type))


def compare_blobs(lhs_blob_file_path, rhs_blob_file_path, np_data_type_str):
    np_data_type = get_np_data_type(np_data_type_str)
    if np_data_type == np.dtype(np.float16):
        return compare_fp16_blobs(lhs_blob_file_path, rhs_blob_file_path)
    elif np_data_type == np.dtype(np.float32):
        return compare_fp32_blobs(lhs_blob_file_path, rhs_blob_file_path)
    return compare_typed_blobs(lhs_blob_file_path, rhs_blob_file_path, np_data_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="nrmse-calculator",
        description="""
Calculate the NRMSE metric for two blobs. NRMSE represents the data similarity.
The valie value of the metric lies in the range [0, 1]
Where:
    0 - means that these two blobs contain completely different binary data
    1 - means that these two blobs has the same binary data, so that they are indistinguishable.
""",
    )
    parser.add_argument("lhs_blob", help="A path to a file containing a first blob for comparison")
    parser.add_argument("rhs_blob", help="A path to another file containing a second blob to compare with the first one")
    parser.add_argument("element_type", help="Precision of blobs binary data")

    args = parser.parse_args()

    print(compare_blobs(args.lhs_blob, args.rhs_blob, args.element_type))
