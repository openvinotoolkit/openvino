# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino as ov
from utils import get_model, get_image

model = get_model()

#! [picture_snippet]
model.reshape([8, 3, 448, 448])
#! [picture_snippet]

#! [set_batch]
model.get_parameters()[0].set_layout(ov.Layout("N..."))
ov.set_batch(model, 5)
#! [set_batch]

#! [simple_spatials_change]
image = get_image()
model.reshape([1, 3, image.shape[0], image.shape[1]])
#! [simple_spatials_change]

#! [obj_to_shape]
port_to_shape = dict()
for input_obj in model.inputs:
    shape = input_obj.get_partial_shape()
    # modify shape to fit your needs
    # ...
    port_to_shape[input_obj] = shape
model.reshape(port_to_shape)
#! [obj_to_shape]

#! [idx_to_shape]
idx_to_shape = dict()
i = 0
for input_obj in model.inputs:
    shape = input_obj.get_partial_shape()
    # modify shape to fit your needs
    # ...
    idx_to_shape[i] = shape
    i += 1
model.reshape(idx_to_shape)
#! [idx_to_shape]

#! [name_to_shape]
name_to_shape = dict()
for input_obj in model.inputs:
    shape = input_obj.get_partial_shape()
    # input may have no name, in such case use map based on input index or port instead
    if len(input_obj.get_names()) != 0:
        # modify shape to fit your needs
        # ...
        name_to_shape[input_obj.get_any_name()] = shape
model.reshape(name_to_shape)
#! [name_to_shape]
