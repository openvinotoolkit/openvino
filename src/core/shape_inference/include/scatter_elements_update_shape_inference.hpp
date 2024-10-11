// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/validation_util.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "utils.hpp"

namespace ov {
namespace op {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const util::ScatterElementsUpdateBase* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4);

    const auto& data_shape = input_shapes[0];
    const auto& indices_shape = input_shapes[1];
    const auto& updates_shape = input_shapes[2];
    const auto& axis_shape = input_shapes[3];

    NODE_VALIDATION_CHECK(op,
                          ov::util::is_rank_compatible_any_of(axis_shape.rank(), {0, 1}),
                          "Axis input shape are required to be scalar or 1D tensor. ",
                          "Got: ",
                          axis_shape);

    const auto& data_rank = data_shape.rank();
    const auto& indices_rank = indices_shape.rank();

    NODE_VALIDATION_CHECK(op,
                          indices_rank.compatible(data_rank),
                          "Indices rank and data rank are required to be equal. ",
                          "Got: ",
                          indices_rank,
                          " and: ",
                          data_rank);

    NODE_VALIDATION_CHECK(op,
                          indices_shape.compatible(updates_shape),
                          "Indices and updates input shapes are required to be equal. ",
                          "Got: ",
                          indices_shape,
                          " and: ",
                          updates_shape);

    if (data_shape.rank().is_static()) {
        if (const auto axis_input = get_input_const_data_as<TShape, int64_t>(op, 3, ta)) {
            ov::util::validate_axis((*axis_input)[0], data_rank, *op);
        }
    }
    return {data_shape};
}

namespace v12 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ScatterElementsUpdate* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    return ov::op::shape_infer(op, input_shapes, ta);
}
}  // namespace v12
}  // namespace op
}  // namespace ov
