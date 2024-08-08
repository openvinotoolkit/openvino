// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/validation_util.hpp"
#include "openvino/op/gather_elements.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v6 {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const GatherElements* op, const std::vector<T>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);
    using DimType = typename T::value_type;

    const auto& data_pshape = input_shapes[0];
    const auto& indices_pshape = input_shapes[1];
    auto data_rank = data_pshape.rank();
    auto indices_rank = indices_pshape.rank();
    auto output_shapes = std::vector<TRShape>();

    NODE_VALIDATION_CHECK(op,
                          indices_rank.is_dynamic() || indices_rank.get_length() >= 1,
                          "indices rank must be >= 1.");
    if (data_rank.is_dynamic()) {
        output_shapes.push_back(indices_pshape);
        return output_shapes;
    } else {
        output_shapes.push_back(data_pshape);
    }
    auto& output_shape = output_shapes[0];

    NODE_VALIDATION_CHECK(op, data_rank.get_length() >= 1, "data rank must be >= 1.");
    const auto axis = ov::util::try_normalize_axis(op->get_axis(), data_rank, *op);

    if (indices_rank.is_dynamic()) {
        // output has the same rank of data
        output_shape[axis] = DimType();
        return output_shapes;
    }

    // left only case when data_rank.is_static() && indices_rank.is_static()
    NODE_VALIDATION_CHECK(op,
                          data_rank.get_length() == indices_rank.get_length(),
                          "data and indices rank must be equal. But instead got: ",
                          data_rank.get_length(),
                          " and ",
                          indices_rank.get_length());

    // if size of the current dimension of indices is unknown it will be retrieved from data
    // e.g., if data_shape = {4, 4, ?}, indices_shape = {1, ?, 5} and axis = 0
    // (and if intervals intersect) then output_pshape will be {1, 4, 5}
    output_shape[axis] = indices_pshape[axis];
    NODE_VALIDATION_CHECK(op,
                          output_shape.merge_into(output_shape, indices_pshape),
                          "Shapes ",
                          data_pshape,
                          " and ",
                          indices_pshape,
                          " are not consistent, `data` and `indices` must have equal or "
                          "intersecting dimensions, except for the dimension at axis index.",
                          axis);
    return output_shapes;
}
}  // namespace v6
}  // namespace op
}  // namespace ov
