// Copyright (C) 2018-2022 Intel Corporationc
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/reduce_modification_for_false_keepdims.hpp"

#include "itt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

bool ModifyReduceBase::is_unsupported_reordered_axes(std::vector<int64_t> reduce_axes,
                                                     size_t num_dim,
                                                     size_t num_spatial) {
    bool is_remains_feature_axis = false;

    // Case to reduce batch axis and spatial axes
    if (reduce_axes.size() > 1 && count(reduce_axes.begin(), reduce_axes.end(), 0) != 0 &&
        count(reduce_axes.begin(), reduce_axes.end(), 1) == 0) {
        is_remains_feature_axis = true;
        // Check if it reduces all spatial axes
        for (size_t idx_spatial = (num_dim - num_spatial); idx_spatial < num_dim; idx_spatial++) {
            if (count(reduce_axes.begin(), reduce_axes.end(), idx_spatial) == 0) {
                is_remains_feature_axis = false;
                break;
            }
        }
    }

    return is_remains_feature_axis;
}

ov::pass::ModifyReduceMeanForFalseKeepDims::ModifyReduceMeanForFalseKeepDims() {
    MATCHER_SCOPE(ModifyReduceMeanForFalseKeepDims);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset10::ReduceMean>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset10::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, modify_reduce_for_false_keepdims<opset10::ReduceMean>());
}

ov::pass::ModifyReduceSumForFalseKeepDims::ModifyReduceSumForFalseKeepDims() {
    MATCHER_SCOPE(ModifyReduceSumForFalseKeepDims);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset10::ReduceSum>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset10::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, modify_reduce_for_false_keepdims<opset10::ReduceSum>());
}

ov::pass::ModifyReduceMaxForFalseKeepDims::ModifyReduceMaxForFalseKeepDims() {
    MATCHER_SCOPE(ModifyReduceMaxForFalseKeepDims);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset10::ReduceMax>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset10::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, modify_reduce_for_false_keepdims<opset10::ReduceMax>());
}

ov::pass::ModifyReduceMinForFalseKeepDims::ModifyReduceMinForFalseKeepDims() {
    MATCHER_SCOPE(ModifyReduceMinForFalseKeepDims);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset10::ReduceMin>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset10::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, modify_reduce_for_false_keepdims<opset10::ReduceMin>());
}

ov::pass::ModifyReduceLogicalAndForFalseKeepDims::ModifyReduceLogicalAndForFalseKeepDims() {
    MATCHER_SCOPE(ModifyReduceLogicalAndForFalseKeepDims);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset10::ReduceLogicalAnd>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset10::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, modify_reduce_for_false_keepdims<opset10::ReduceLogicalAnd>());
}

ov::pass::ModifyReduceLogicalOrForFalseKeepDims::ModifyReduceLogicalOrForFalseKeepDims() {
    MATCHER_SCOPE(ModifyReduceLogicalOrForFalseKeepDims);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset10::ReduceLogicalOr>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset10::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, modify_reduce_for_false_keepdims<opset10::ReduceLogicalOr>());
}
