// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/util/scatter_nd_base.hpp>

#include "utils.hpp"

template <class T>
void shape_infer(const ov::op::util::ScatterNDBase* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 && output_shapes.size() == 1);
    const auto& inputs_shape = input_shapes[0];
    const auto& indices_shape = input_shapes[1];
    const auto& updates_shape = input_shapes[2];

    const auto& inputs_rank = inputs_shape.rank();
    const auto& indices_rank = indices_shape.rank();
    const auto& updates_rank = updates_shape.rank();

    NODE_VALIDATION_CHECK(op,
                          indices_rank != 0 && inputs_rank != 0,
                          "Indices rank and inputs_rank are expected to be at least 1");

    NODE_VALIDATION_CHECK(
        op,
        inputs_rank.is_dynamic() || indices_rank.is_dynamic() || indices_shape[indices_shape.size() - 1].is_dynamic() ||
            static_cast<size_t>(indices_shape[indices_shape.size() - 1].get_length()) <= inputs_shape.size(),
        "Last dimension of indices can be at most the rank of inputs");

    if (inputs_rank.is_static() && indices_rank.is_static() && updates_rank.is_static() &&
        indices_shape[indices_shape.size() - 1].is_static()) {
        auto expected_updates_rank =
            indices_shape.size() + inputs_shape.size() - indices_shape[indices_shape.size() - 1].get_length() - 1;
        // If expected updates rank is 0D it also can be a tensor with one element
        NODE_VALIDATION_CHECK(op,
                              updates_shape.size() == expected_updates_rank || expected_updates_rank == 0,
                              "Rank of updates must be rank of inputs + rank of indices - last dimension of indices "
                              "- 1");

        bool compatible = true;
        size_t static_indices_rank = indices_shape.size();
        for (size_t i = 0; i < static_indices_rank - 1; i++) {
            compatible = compatible && updates_shape[i].compatible(indices_shape[i]);
            NODE_VALIDATION_CHECK(op, compatible, "updates_shape[0:indices_rank-1] shape must be indices_shape[:-1]");
        }
        size_t j = indices_shape[static_indices_rank - 1].get_length();
        for (int64_t i = static_indices_rank - 1; i < static_cast<int64_t>(expected_updates_rank); i++, j++) {
            compatible = compatible && updates_shape[i].compatible(inputs_shape[j]);
            NODE_VALIDATION_CHECK(op,
                                  compatible,
                                  "updates_shape[indices_rank-1:] shape must be input_shape[indices_shape[-1]:]");
        }
    }
    output_shapes[0] = inputs_shape;
}
