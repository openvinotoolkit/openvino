# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def batched_common_infer_step(device, **additional_args):
    # return 'reshape' method for setting batch because old 'batch_size' method still not implemented in new API
    return "infer", {"ie_sync": {"device": device},
                     **additional_args}


def common_infer_step(device, **additional_args):
    return "infer", {"ie_sync": {"device": device,
                                 "network_modifiers": {},
                                 **additional_args}}


def batch_reshape_infer_step(device, **additional_args):
    return "infer", {"ie_sync": {"device": device},
                     **additional_args}


def reshape_input_shape_infer_step(device, input_file_path, **additional_args):
    return "infer", {"ie_sync": {"device": device,
                                 "network_modifiers":
                                     {"reshape_input_shape": {"input_path": input_file_path}},
                                 **additional_args}}


def cpu_extension_infer_step(device, **additional_args):
    return "infer", {"ie_sync": {"device": device, "cpu_extension": "cpu_extension"},
                     **additional_args}


def sequence_infer_step(device, **additional_args):
    # return 'reshape' method for setting batch because old 'batch_size' method still not implemented in new API
    return "infer", {"ie_sequence": {"device": device},
                     **additional_args}
