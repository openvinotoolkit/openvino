# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def read_npz_input(path):
    return "read_input", {"npz": {"path": path}}


def read_npy_input(path):
    return "read_input", {"npy": {"inputs_map": path}}


def read_ark_input(path):
    return "read_input", {"ark": {"inputs_map": path}}


def read_img_input(path):
    return "read_input", {"img": {"inputs_map": path}}


def read_external_input(path):
    return "read_input", {"external_data": {"data": path}}


def read_pb_input(path):
    return "read_input", {"pb": {"inputs_map": path}}


def read_pt_input(path):
    return "read_input", {"pt": {"path": path}}


def generate_tf_hub_inputs(model):
    return {"read_input": {"generate_tf_hub_inputs": {"model": model}}}
