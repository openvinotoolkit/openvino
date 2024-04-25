// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_reduce_to_reshape.hpp"

#include "itt.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

bool CvtReduceBase::is_redundant(ov::Shape input, ov::Shape output) {
    if (shape_size(input) != shape_size(output))
        return false;

    return true;
}

ov::pass::ConvertReduceMeanToReshape::ConvertReduceMeanToReshape() {
    MATCHER_SCOPE(ConvertReduceMeanToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<ov::op::v1::ReduceMean>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<ov::op::v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<ov::op::v1::ReduceMean>());
}

ov::pass::ConvertReduceSumToReshape::ConvertReduceSumToReshape() {
    MATCHER_SCOPE(ConvertReduceSumToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<ov::op::v1::ReduceSum>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<ov::op::v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<ov::op::v1::ReduceSum>());
}

ov::pass::ConvertReduceProdToReshape::ConvertReduceProdToReshape() {
    MATCHER_SCOPE(ConvertReduceProdToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<ov::op::v1::ReduceProd>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<ov::op::v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<ov::op::v1::ReduceProd>());
}

ov::pass::ConvertReduceMaxToReshape::ConvertReduceMaxToReshape() {
    MATCHER_SCOPE(ConvertReduceMaxToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<ov::op::v1::ReduceMax>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<ov::op::v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<ov::op::v1::ReduceMax>());
}

ov::pass::ConvertReduceMinToReshape::ConvertReduceMinToReshape() {
    MATCHER_SCOPE(ConvertReduceMinToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<ov::op::v1::ReduceMin>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<ov::op::v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<ov::op::v1::ReduceMin>());
}

ov::pass::ConvertReduceLogicalAndToReshape::ConvertReduceLogicalAndToReshape() {
    MATCHER_SCOPE(ConvertReduceLogicalAndToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<ov::op::v1::ReduceLogicalAnd>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<ov::op::v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<ov::op::v1::ReduceLogicalAnd>());
}

ov::pass::ConvertReduceLogicalOrToReshape::ConvertReduceLogicalOrToReshape() {
    MATCHER_SCOPE(ConvertReduceLogicalOrToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<ov::op::v1::ReduceLogicalOr>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<ov::op::v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<ov::op::v1::ReduceLogicalOr>());
}
