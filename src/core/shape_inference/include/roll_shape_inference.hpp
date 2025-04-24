// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/validation_util.hpp"
#include "openvino/op/roll.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v7 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Roll* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3);

    const auto& data_pshape = input_shapes[0];
    const auto& shift_pshape = input_shapes[1];
    const auto& axes_pshape = input_shapes[2];

    if (shift_pshape.rank().is_static()) {
        const auto& shift_rank = shift_pshape.size();
        NODE_VALIDATION_CHECK(op, shift_rank <= 1, "Shift must be a scalar or 1D tensor.");
        // If shift is a scalar, than axes can be arbitrary 1d tensor and we don't need
        // to check shift shape consistency with axes, otherwise the check is needed.
        if (shift_rank == 1) {
            NODE_VALIDATION_CHECK(op,
                                  shift_pshape.compatible(axes_pshape),
                                  "If shift is a 1D vector, axes must be a 1D tensor of the same size.");
        }
    }

    NODE_VALIDATION_CHECK(op,
                          axes_pshape.rank().is_dynamic() || axes_pshape.size() <= 1,
                          "Axes must be a scalar or 1D tensor.");

    if (data_pshape.rank().is_static()) {
        if (auto axes = get_input_const_data_as<TRShape, int64_t>(op, 2, ta)) {
            ov::util::validate_axes(*axes, data_pshape.rank(), *op);
        }
    }

    return {data_pshape};
}
}  // namespace v7
}  // namespace op
}  // namespace ov
