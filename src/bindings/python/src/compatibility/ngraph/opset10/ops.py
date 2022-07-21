# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all openvino ops."""
from functools import partial
from typing import List, Optional

from ngraph.impl import Node
from ngraph.opset_utils import _get_node_factory
from ngraph.utils.decorators import nameable_op
from ngraph.utils.types import (
    NodeInput,
    as_nodes,
    as_node,
    make_constant_node,
)

_get_node_factory_opset4 = partial(_get_node_factory, "opset4")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def interpolate(
    image: NodeInput,
    output_shape: NodeInput,
    scales: NodeInput,
    mode: str,
    shape_calculation_mode: str,
    pads_begin: Optional[List[int]] = None,
    pads_end: Optional[List[int]] = None,
    coordinate_transformation_mode: str = "half_pixel",
    nearest_mode: str = "round_prefer_floor",
    antialias: bool = False,
    cube_coeff: float = -0.75,
    axes: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Perform interpolation of independent slices in input tensor.

    :param  image:         The node providing input tensor with data for interpolation.
    :param  output_shape:  1D tensor describing output shape for spatial axes.
    :param  scales:        1D tensor describing scales for spatial axes.
    :param  mode:          Specifies type of interpolation. Possible values are: nearest, linear,
                           linear_onnx, cubic.
    :param  shape_calculation_mode:
                           Specifies which input, sizes or scales, is used to calculate an output
                           shape.
    :param  pads_begin:    Specifies the number of pixels to add to the beginning of the image
                           being interpolated. Default is None.
    :param  pads_end:      Specifies the number of pixels to add to the end of the image being
                           interpolated. Default is None.
    :param  coordinate_transformation_mode:
                           Specifies how to transform the coordinate in the resized tensor to the
                           coordinate in the original tensor. Default is "half_pixel".
    :param  nearest_mode:  Specifies round mode when mode == nearest and is used only when
                           mode == nearest. Default is "round_prefer_floor".
    :param  antialias:     Specifies whether to perform anti-aliasing. Default is False.
    :param  cube_coeff:    Specifies the parameter a for cubic interpolation. Default is -0.75.
    :param  axes:          1D tensor specifying dimension indices where interpolation is applied.
                           Default is None.
    :param  name:          Optional name for the output node. Default is None.
    :return: Node representing interpolation operation.
    """
    attrs = {
        "mode": mode,
        "shape_calculation_mode": shape_calculation_mode,
        "coordinate_transformation_mode": coordinate_transformation_mode,
        "nearest_mode": nearest_mode,
        "antialias": antialias,
        "cube_coeff": cube_coeff,
    }

    attrs["pads_begin"] = [] if pads_begin is None else pads_begin
    attrs["pads_end"] = [] if pads_end is None else pads_end

    inputs = as_nodes(image, output_shape, scales) if axes is None else as_nodes(image,
                                                                                 output_shape,
                                                                                 scales, axes)

    # This is an update of the operator version, so even though this is opset 10,
    # the operator is taken from opset 4.
    return _get_node_factory_opset4().create("Interpolate", inputs, attrs)
