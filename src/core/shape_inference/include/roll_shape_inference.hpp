// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/roll.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v7 {

template <class T>
void shape_infer(const ov::op::v7::Roll* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 && output_shapes.size() == 1);

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

    if (axes_pshape.rank().is_static()) {
        const auto& axes_rank = axes_pshape.size();
        NODE_VALIDATION_CHECK(op, axes_rank <= 1, "Axes must be a scalar or 1D tensor.");
    }

    std::vector<int64_t> axes{};

    if (get_data_as_int64<T>(2, op, axes, constant_data)) {
        if (data_pshape.rank().is_static()) {
            const auto& data_rank = data_pshape.size();
            for (int64_t& axis : axes) {
                NODE_VALIDATION_CHECK(op,
                                      axis < static_cast<int64_t>(data_rank),
                                      "Axes must be less than data tensor rank. Got "
                                      "data tensor rank: ",
                                      data_rank,
                                      ", axis: ",
                                      axis);
                if (axis < 0) {
                    axis += static_cast<int64_t>(data_rank);
                }
                NODE_VALIDATION_CHECK(op,
                                      axis >= 0,
                                      "Axes must be positive or equal to zero. Got "
                                      "axis: ",
                                      axis);
            }
        }
    }

    output_shapes[0] = input_shapes[0];
}

}  // namespace v7
}  // namespace op
}  // namespace ov