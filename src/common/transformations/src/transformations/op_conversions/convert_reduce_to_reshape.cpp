// Copyright (C) 2018-2026 Intel Corporation
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

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

bool CvtReduceBase::is_redundant(ov::Shape input, ov::Shape output) {
    if (ov::shape_size(input) != ov::shape_size(output))
        return false;

    return true;
}

namespace ov::pass {

ConvertReduceMeanToReshape::ConvertReduceMeanToReshape() {
    MATCHER_SCOPE(ConvertReduceMeanToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<v1::ReduceMean>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<v1::ReduceMean>());
}

ConvertReduceSumToReshape::ConvertReduceSumToReshape() {
    MATCHER_SCOPE(ConvertReduceSumToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<v1::ReduceSum>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<v1::ReduceSum>());
}

ConvertReduceProdToReshape::ConvertReduceProdToReshape() {
    MATCHER_SCOPE(ConvertReduceProdToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<v1::ReduceProd>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<v1::ReduceProd>());
}

ConvertReduceMaxToReshape::ConvertReduceMaxToReshape() {
    MATCHER_SCOPE(ConvertReduceMaxToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<v1::ReduceMax>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<v1::ReduceMax>());
}

ConvertReduceMinToReshape::ConvertReduceMinToReshape() {
    MATCHER_SCOPE(ConvertReduceMinToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<v1::ReduceMin>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<v1::ReduceMin>());
}

ConvertReduceLogicalAndToReshape::ConvertReduceLogicalAndToReshape() {
    MATCHER_SCOPE(ConvertReduceLogicalAndToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<v1::ReduceLogicalAnd>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<v1::ReduceLogicalAnd>());
}

ConvertReduceLogicalOrToReshape::ConvertReduceLogicalOrToReshape() {
    MATCHER_SCOPE(ConvertReduceLogicalOrToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<v1::ReduceLogicalOr>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<v1::ReduceLogicalOr>());
}

}  // namespace ov::pass
