// Copyright (C) 2018-2022 Intel Corporationconvert_reduce_to_pooling
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_reduce_to_reshape.hpp"

#include "itt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

bool CvtReduceBase::is_redundant(ngraph::Shape input, ngraph::Shape output) {
    if (shape_size(input) != shape_size(output))
        return false;

    for (size_t idx = 0; idx < input.size(); idx++) {
        if (input[idx] != output[idx] && input[idx] != 1)
            return false;
    }

    return true;
}

ov::pass::ConvertReduceMeanToReshape::ConvertReduceMeanToReshape() {
    MATCHER_SCOPE(ConvertReduceMeanToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset1::ReduceMean>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset1::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<opset1::ReduceMean>());
}

ov::pass::ConvertReduceSumToReshape::ConvertReduceSumToReshape() {
    MATCHER_SCOPE(ConvertReduceSumToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset1::ReduceSum>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset1::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<opset1::ReduceSum>());
}

ov::pass::ConvertReduceMaxToReshape::ConvertReduceMaxToReshape() {
    MATCHER_SCOPE(ConvertReduceMaxToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset1::ReduceMax>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset1::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<opset1::ReduceMax>());
}

ov::pass::ConvertReduceMinToReshape::ConvertReduceMinToReshape() {
    MATCHER_SCOPE(ConvertReduceMinToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset1::ReduceMin>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset1::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<opset1::ReduceMin>());
}

ov::pass::ConvertReduceLogicalAndToReshape::ConvertReduceLogicalAndToReshape() {
    MATCHER_SCOPE(ConvertReduceLogicalAndToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset1::ReduceLogicalAnd>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset1::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<opset1::ReduceLogicalAnd>());
}

ov::pass::ConvertReduceLogicalOrToReshape::ConvertReduceLogicalOrToReshape() {
    MATCHER_SCOPE(ConvertReduceLogicalOrToReshape);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<opset1::ReduceLogicalOr>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<opset1::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_reshape<opset1::ReduceLogicalOr>());
}
