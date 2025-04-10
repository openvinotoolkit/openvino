// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/rms_norm.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace internal {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const RMSNorm* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    const auto inputs_count = input_shapes.size();
    const auto has_scale_input = inputs_count == 3;
    NODE_SHAPE_INFER_CHECK(op, input_shapes, inputs_count == 2 || has_scale_input);

    const auto& data_shape = input_shapes[0];
    const auto& data_rank = data_shape.rank();
    const auto& axes_shape = input_shapes[1];
    const auto& axes_rank = axes_shape.rank();

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           ov::util::is_rank_compatible_any_of(axes_rank, {0, 1}),
                           "Axes input must be a scalar or 1D input. Got: ",
                           axes_shape);

    // Further validation requires data rank to be static
    if (data_rank.is_dynamic()) {
        return {data_shape};
    }

    if (axes_shape.rank().is_static()) {
        const bool has_axes_compatible = axes_shape.size() == 0 || axes_shape[0].is_dynamic() ||
                                         cmp::ge(data_rank.get_length(), axes_shape.get_shape()[0]);
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               has_axes_compatible,
                               "Number of the axes can't be higher than the rank of the data shape.");
    }

    if (has_scale_input) {  // Validate scale input
        TRShape scale_shape = input_shapes[2];
        const bool is_scale_shape_broadcastable =
            TRShape::broadcast_merge_into(scale_shape, data_shape, ov::op::AutoBroadcastType::NUMPY);
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               is_scale_shape_broadcastable,
                               "Scale input shape must be broadcastable to the shape of the data input.");
    }

    // Axes values validation
    if (data_rank.is_static()) {
        if (auto axes_val = ov::op::get_input_const_data_as<TRShape, int64_t>(op, 1, tensor_accessor)) {
            ov::util::validate_axes(*axes_val, data_rank, *op);
        }
    }

    return {data_shape};
}
}  // namespace internal
}  // namespace op
}  // namespace ov
