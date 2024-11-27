# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino


# Creates a new file with a given name, populates it with data from a given Constant,
# returns a new Constant node with content memory-mapped to that file.
# Doesn't remove the file in the end of the returned Constant's life time.
def move_constant_to_file(constant, path):
    openvino.save_tensor_data(constant.get_tensor_view(), path)
    mmapped = openvino.read_tensor_data(path, constant.get_output_element_type(0), constant.get_output_partial_shape(0))
    return openvino.runtime.op.Constant(mmapped, shared_memory=True)
