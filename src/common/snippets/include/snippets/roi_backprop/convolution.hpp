// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/gather.hpp>
#include <openvino/core/partial_shape.hpp>

namespace ov {
namespace op {
namespace v1 {
template <class ShapeType>
void roi_backprop(
        const Convolution* op,
        const std::vector<ShapeType>& input_shapes,
        std::vector<ShapeType>& roi_shapes,
        std::vector<ov::Shape>& strides) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2ul) && (roi_shapes.size() == 1));

    // TODO: just to test
    //
}

}  // namespace v1
}  // namespace op
}  // namespace ov