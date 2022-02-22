# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def format_input_batch(input_batch, idx):
    if isinstance(input_batch[0], tuple):
        annotation_data, input_data = input_batch[0], input_batch[1]
        formatted_batch = [annotation_data, input_data]
    else:
        input_data, annotation = input_batch[0], input_batch[1]
        formatted_batch = [(idx, annotation), input_data]

    if len(input_batch) == 3:
        meta_data = input_batch[2]
        formatted_batch.append(meta_data)
    return tuple(formatted_batch)