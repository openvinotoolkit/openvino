// Copyright (C) 2018-2022 Intel Corporationc
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/reduce_modification_add_transpose.hpp"

#include "itt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

std::vector<uint16_t> ModifyReduceBaseTranspose::get_reverse_order(std::vector<uint16_t>& transposed_order) {
    auto original_order = transposed_order;

    for (size_t idx = 0; idx < transposed_order.size(); idx++) {
        original_order.at(transposed_order[idx]) = static_cast<uint16_t>(idx);
    }

    return original_order;
}

bool ModifyReduceBaseTranspose::is_reduce_single_spatial_axis(const std::vector<int64_t>& reduce_axes,
                                                              size_t num_dim,
                                                              size_t num_spatial,
                                                              std::vector<uint16_t>& order) {
    bool is_reduced_spatial = false;
    size_t reduced_cnt = 0;
    std::vector<uint16_t> transposed_order;

    transposed_order.push_back(0);
    if (count(reduce_axes.begin(), reduce_axes.end(), 1) != 0) {
        std::vector<uint16_t> reduced_axis;
        for (size_t idx_spatial = (num_dim - num_spatial); idx_spatial < num_dim; idx_spatial++) {
            if (count(reduce_axes.begin(), reduce_axes.end(), idx_spatial) == 0) {
                is_reduced_spatial = true;
                transposed_order.push_back(static_cast<uint16_t>(idx_spatial));
                transposed_order.push_back(1);
            } else {
                reduced_cnt++;
                reduced_axis.push_back(static_cast<uint16_t>(idx_spatial));
            }
        }

        for (auto axis : reduced_axis)
            transposed_order.push_back(axis);
    }

    // one spatial axis is un-reduced (will be moved to feature axis)
    if (is_reduced_spatial && reduced_cnt == 1) {
        order = transposed_order;
        return true;
    } else {
        return false;
    }
}

ov::pass::ModifyReduceMeanToAddTranspose::ModifyReduceMeanToAddTranspose() {
    MATCHER_SCOPE(ModifyReduceMeanToAddTranspose);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset1::ReduceMean>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset1::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, add_transpose_to_reduce<opset1::ReduceMean>());
}

ov::pass::ModifyReduceSumToAddTranspose::ModifyReduceSumToAddTranspose() {
    MATCHER_SCOPE(ModifyReduceSumToAddTranspose);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset1::ReduceSum>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset1::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, add_transpose_to_reduce<opset1::ReduceSum>());
}

ov::pass::ModifyReduceMaxToAddTranspose::ModifyReduceMaxToAddTranspose() {
    MATCHER_SCOPE(ModifyReduceMaxToAddTranspose);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset1::ReduceMax>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset1::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, add_transpose_to_reduce<opset1::ReduceMax>());
}

ov::pass::ModifyReduceMinToAddTranspose::ModifyReduceMinToAddTranspose() {
    MATCHER_SCOPE(ModifyReduceMinToAddTranspose);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset1::ReduceMin>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset1::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, add_transpose_to_reduce<opset1::ReduceMin>());
}

ov::pass::ModifyReduceLogicalAndToAddTranspose::ModifyReduceLogicalAndToAddTranspose() {
    MATCHER_SCOPE(ModifyReduceLogicalAndToAddTranspose);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset1::ReduceLogicalAnd>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset1::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, add_transpose_to_reduce<opset1::ReduceLogicalAnd>());
}

ov::pass::ModifyReduceLogicalOrToAddTranspose::ModifyReduceLogicalOrToAddTranspose() {
    MATCHER_SCOPE(ModifyReduceLogicalOrToAddTranspose);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset1::ReduceLogicalOr>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset1::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, add_transpose_to_reduce<opset1::ReduceLogicalOr>());
}
