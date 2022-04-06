// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/gather_elements.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v6 {
template <class T>
void shape_infer(const GatherElements* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;

    const auto& data_pshape = input_shapes[0];
    const auto& indices_pshape = input_shapes[1];
    auto data_rank = data_pshape.rank();
    auto indices_rank = indices_pshape.rank();
    auto& output_shape = output_shapes[0];

    int64_t axis = op->m_axis;
    if (data_rank.is_static())
        axis = ov::normalize_axis(op, axis, data_rank);

    output_shape = indices_pshape;

    NODE_VALIDATION_CHECK(op, data_rank.is_dynamic() || data_rank.get_length() >= 1, "data rank must be >= 1.");

    NODE_VALIDATION_CHECK(op,
                          indices_rank.is_dynamic() || indices_rank.get_length() >= 1,
                          "indices rank must be >= 1.");

    if (data_rank.is_static() && indices_rank.is_dynamic()) {
        // output has the same rank of data
        output_shape = data_pshape;
        output_shape[axis] = DimType();
        return;
    }

    if (data_rank.is_dynamic()) {
        // can't decide rank, set it to all dynamic
        if (indices_rank.is_dynamic())
            output_shape = PartialShape::dynamic();
        return;
    }

    // left only case when data_rank.is_static() && indices_rank.is_static()
    NODE_VALIDATION_CHECK(op,
                          data_rank.get_length() == indices_rank.get_length(),
                          "data and indices rank must be equal. But instead got: ",
                          data_rank.get_length(),
                          " and ",
                          indices_rank.get_length());

    for (int i = 0; i < indices_rank.get_length(); i++) {
        if (i != axis) {
            // if size of the current dimension of indices is unknown it will be retrieved from data
            // e.g., if data_shape = {4, 4, ?}, indices_shape = {1, ?, 5} and axis = 0
            // (and if intervals intersect) then output_pshape will be {1, 4, 5}

            NODE_VALIDATION_CHECK(op,
                                  data_pshape[i].compatible(indices_pshape[i]),
                                  "Shapes ",
                                  data_pshape,
                                  " and ",
                                  indices_pshape,
                                  " are not consistent. data and indices must have equal or "
                                  "intersecting sizes, except for axis ",
                                  axis);

            output_shape[i] = data_pshape[i] & indices_pshape[i];
        }
    }
}
}  // namespace v6
}  // namespace op
}  // namespace ov