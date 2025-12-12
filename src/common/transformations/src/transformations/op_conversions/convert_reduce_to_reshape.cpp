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


using ov::pass::pattern::any_input;
using ov::pass::pattern::wrap_type;
using ov::pass::pattern::Matcher;
using ov::pass::pattern::has_static_shape;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
bool CvtReduceBase::is_redundant(ov::Shape input, ov::Shape output) {
    if (shape_size(input) != shape_size(output))
        return false;

    return true;
}

ov::pass::ConvertReduceMeanToReshape::ConvertReduceMeanToReshape() {
    MATCHER_SCOPE(ConvertReduceMeanToReshape);
    auto m = std::make_shared<Matcher>(
        wrap_type<v1::ReduceMean>(
            {any_input(has_static_shape()),
             wrap_type<v0::Constant>()},
            has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<v1::ReduceMean>());
}

ov::pass::ConvertReduceSumToReshape::ConvertReduceSumToReshape() {
    MATCHER_SCOPE(ConvertReduceSumToReshape);
    auto m = std::make_shared<Matcher>(
        wrap_type<v1::ReduceSum>(
            {any_input(has_static_shape()),
             wrap_type<v0::Constant>()},
            has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<v1::ReduceSum>());
}

ov::pass::ConvertReduceProdToReshape::ConvertReduceProdToReshape() {
    MATCHER_SCOPE(ConvertReduceProdToReshape);
    auto m = std::make_shared<Matcher>(
        wrap_type<v1::ReduceProd>(
            {any_input(has_static_shape()),
             wrap_type<v0::Constant>()},
            has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<v1::ReduceProd>());
}

ov::pass::ConvertReduceMaxToReshape::ConvertReduceMaxToReshape() {
    MATCHER_SCOPE(ConvertReduceMaxToReshape);
    auto m = std::make_shared<Matcher>(
        wrap_type<v1::ReduceMax>(
            {any_input(has_static_shape()),
             wrap_type<v0::Constant>()},
            has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<v1::ReduceMax>());
}

ov::pass::ConvertReduceMinToReshape::ConvertReduceMinToReshape() {
    MATCHER_SCOPE(ConvertReduceMinToReshape);
    auto m = std::make_shared<Matcher>(
        wrap_type<v1::ReduceMin>(
            {any_input(has_static_shape()),
             wrap_type<v0::Constant>()},
            has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<v1::ReduceMin>());
}

ov::pass::ConvertReduceLogicalAndToReshape::ConvertReduceLogicalAndToReshape() {
    MATCHER_SCOPE(ConvertReduceLogicalAndToReshape);
    auto m = std::make_shared<Matcher>(
        wrap_type<v1::ReduceLogicalAnd>(
            {any_input(has_static_shape()),
             wrap_type<v0::Constant>()},
            has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<v1::ReduceLogicalAnd>());
}

ov::pass::ConvertReduceLogicalOrToReshape::ConvertReduceLogicalOrToReshape() {
    MATCHER_SCOPE(ConvertReduceLogicalOrToReshape);
    auto m = std::make_shared<Matcher>(
        wrap_type<v1::ReduceLogicalOr>(
            {any_input(has_static_shape()),
             wrap_type<v0::Constant>()},
            has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<v1::ReduceLogicalOr>());
}
