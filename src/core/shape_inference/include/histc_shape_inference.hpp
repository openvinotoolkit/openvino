// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/op/histc.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v17 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Histc* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1, "Histc must have exactly one input.");

    using TDim = typename TRShape::value_type;
    return {TRShape{static_cast<TDim>(op->get_bins())}};
}

}  // namespace v17
}  // namespace op
}  // namespace ov
