// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const Unsqueeze* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op,
                          input_shapes.size() == Unsqueeze::IN_COUNT && output_shapes.size() == Unsqueeze::OUT_COUNT);
    const auto& arg_shape = input_shapes[Unsqueeze::ARG];
    auto& out_shape = output_shapes[Unsqueeze::OUT];

    std::vector<int64_t> axes_val;
    const auto has_axes = get_data_as_int64<T>(Unsqueeze::AXES, op, axes_val, constant_data);

    if (has_axes && arg_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op, !axes_val.empty(), "'axes' input is mandatory");
        // Remove repeated axes on input
        std::unordered_set<int64_t> tmp(std::make_move_iterator(axes_val.begin()),
                                        std::make_move_iterator(axes_val.end()));
        std::vector<int64_t> unique_axes(std::make_move_iterator(tmp.begin()), std::make_move_iterator(tmp.end()));

        const auto expanded_rank = arg_shape.rank().get_length() + unique_axes.size();

        // Normalize then remove repeated axes after normalization.
        normalize_axes(op, expanded_rank, unique_axes);
        const std::set<int64_t> axes(std::make_move_iterator(unique_axes.begin()),
                                     std::make_move_iterator(unique_axes.end()));

        out_shape = arg_shape;
        for (const auto& axis : axes) {
            NODE_VALIDATION_CHECK(op, axis <= out_shape.size() + 1, "provided 'axes' value ", axis, " is not valid.");
            // As shape not throw exception on repeated axis it has to be check if insert or append dimension.
            // This will be not required if this op has same behaviour as numpy expand_dims.
            if (axis <= out_shape.size()) {
                out_shape.insert(std::next(std::begin(out_shape), axis), 1);
            } else {
                // Append dimension at end when there is difference in size of input axes and after normalization
                // e.g. input shape {2,3,4} axes_value(4,-1) then output rank is determined as 5,
                // but after final normalization and removing duplicates it points sam location in shape.
                // The numpy throws exception "repeated axis" in that case.
                out_shape.push_back(1);
            }
        }
    } else {
        out_shape = ov::PartialShape::dynamic();
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
