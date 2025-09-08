// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/util/scatter_nd_base.hpp>

#include "utils.hpp"

namespace ov {
namespace op {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const util::ScatterNDBase* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3);
    const auto& inputs_shape = input_shapes[util::ScatterNDBase::INPUTS];
    const auto& indices_shape = input_shapes[util::ScatterNDBase::INDICES];
    const auto& updates_shape = input_shapes[util::ScatterNDBase::UPDATES];

    const auto& inputs_rank = inputs_shape.rank();
    const auto& indices_rank = indices_shape.rank();
    const auto& updates_rank = updates_shape.rank();

    NODE_VALIDATION_CHECK(op,
                          indices_rank != 0 && inputs_rank != 0,
                          "Indices rank and inputs_rank are expected to be at least 1");

    if (inputs_rank.is_static() && indices_rank.is_static()) {
        const auto last_idx_pos = indices_shape.size() - 1;
        const auto& last_idx_dim = indices_shape[last_idx_pos];

        if (last_idx_dim.is_static()) {
            const auto last_idx_dim_size = static_cast<size_t>(last_idx_dim.get_length());

            NODE_VALIDATION_CHECK(op,
                                  last_idx_dim_size <= inputs_shape.size(),
                                  "Last dimension of indices can be at most the rank of inputs");
            if (updates_rank.is_static()) {
                // Used last_idx_pos because is equal rank of indices - 1
                const auto expected_updates_rank = inputs_shape.size() + last_idx_pos - last_idx_dim_size;
                // If expected updates rank is 0D it also can be a tensor with one element
                NODE_VALIDATION_CHECK(
                    op,
                    updates_shape.size() == expected_updates_rank || expected_updates_rank == 0,
                    "Rank of updates must be rank of inputs + rank of indices - last dimension of indices - 1");

                auto update_iter = updates_shape.begin();
                auto is_update_compatible = [&update_iter](const typename TShape::value_type& d) -> bool {
                    return d.compatible(*update_iter++);
                };

                NODE_VALIDATION_CHECK(
                    op,
                    std::all_of(indices_shape.begin(), indices_shape.begin() + last_idx_pos, is_update_compatible),
                    "updates_shape[0:indices_rank-1] shape must be indices_shape[:-1]");

                NODE_VALIDATION_CHECK(
                    op,
                    std::all_of(inputs_shape.begin() + last_idx_dim_size, inputs_shape.end(), is_update_compatible),
                    "updates_shape[indices_rank-1:] shape must be input_shape[indices_shape[-1]:]");
            }
        }
    }

    return {inputs_shape};
}
}  // namespace op
}  // namespace ov
