# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino

# Creates a new file with a given name, populates it with data from a given Constant, returns a new Constant node with data allocated in mmapped region from that file
# Doesn't remove the file in the end of life.
def move_constant_to_file(constant, path):
    openvino.save_tensor_data(openvino.Tensor(constant.data), path)  # FIXME Data copying, use get_tensor_view when it becomes available
    mmapped = openvino.read_tensor_data(path, constant.get_output_element_type(0), constant.get_output_partial_shape(0))
    return openvino.runtime.op.Constant(mmapped)