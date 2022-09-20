// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/transpose.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v1 {
template <typename Rank>
bool is_valid_permutation(ov::AxisVector permutation, Rank rank) {
    std::vector<bool> axis_occurs(permutation.size(), false);

    // Check bounds if rank is static
    if (rank.is_static()) {
        auto bound = rank.get_length();
        for (auto axis : permutation) {
            if (static_cast<decltype(bound)>(axis) >= bound) {
                return false;
            }
        }
    }

    for (auto& axis : permutation) {
        axis_occurs[axis] = true;
    }

    for (size_t axis = 0; axis < permutation.size(); axis++) {
        if (!axis_occurs[axis]) {
            return false;
        }
    }

    return (rank.is_dynamic() || static_cast<int64_t>(permutation.size()) == rank.get_length());
}

template <typename T>
T apply_permutation(T input, AxisVector order) {

    T output;
    output.resize(input.size());

    for (size_t i = 0; i < order.size(); i++) {
        output[i] = input.at(order.at(i));
    }

    return output;
}

template <class T>
void shape_infer(const Transpose* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    const auto& input_order_shape = input_shapes[1];
    NODE_VALIDATION_CHECK(op, input_order_shape.rank().compatible(1), "Input order must be a vector.");

    const auto& arg_shape = input_shapes[0];

    if (arg_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(
            op,
            input_order_shape.compatible(T{arg_shape.rank().get_length()}) ||
                (input_order_shape.is_static() && input_order_shape.size() == 1 && input_order_shape[0] == 0),
            "Input order must have shape [n], where n is the rank of arg.");
    } else {
        auto output_rank = arg_shape.rank();
        if (output_rank.is_dynamic() && input_order_shape.is_static() && input_order_shape[0].get_length())
            output_rank = input_order_shape[0].get_length();
        output_shapes[0] = ov::PartialShape::dynamic(output_rank);
    }

    if (const auto& input_const = get_constant_from_source(op->input_value(1))) {
        auto permutation = input_const->get_axis_vector_val();
        if (permutation.empty()) {
            for (int64_t i = 1; i <= arg_shape.rank().get_length(); ++i)
                permutation.emplace_back(arg_shape.rank().get_length() - i);
        }
        NODE_VALIDATION_CHECK(op,
                              is_valid_permutation(permutation, arg_shape.rank()),
                              "Permutation ",
                              permutation,
                              " is not valid for input shape ",
                              arg_shape);
        output_shapes[0] = apply_permutation(arg_shape, permutation);
    }
}

}  // namespace v0
}  // namespace op
}  // namespace ov