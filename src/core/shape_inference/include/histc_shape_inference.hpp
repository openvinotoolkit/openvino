// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/histc.hpp"
#include "utils.hpp"

namespace ov::op::v17 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Histc* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);
    NODE_VALIDATION_CHECK(op, op->get_bins() > 0, "Attribute 'bins' must be positive. Got: ", op->get_bins());
    return {TRShape{static_cast<typename TRShape::value_type>(op->get_bins())}};
}
}  // namespace ov::op::v17
