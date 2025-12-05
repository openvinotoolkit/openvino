// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"

#include "itt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ConvertReduceMeanToPooling::ConvertReduceMeanToPooling() {
    MATCHER_SCOPE(ConvertReduceMeanToPooling);
    auto m = std::make_shared<ov::pass::pattern::Matcher>(
        ov::pass::pattern::wrap_type<ov::op::v1::ReduceMean>(
            {ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape()), ov::pass::pattern::wrap_type<ov::op::v0::Constant>()},
            ov::pass::pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_pooling<ov::op::v1::ReduceMean>());
}
ov::pass::ConvertReduceMaxToPooling::ConvertReduceMaxToPooling() {
    MATCHER_SCOPE(ConvertReduceMaxToPooling);
    auto m = std::make_shared<ov::pass::pattern::Matcher>(
        ov::pass::pattern::wrap_type<ov::op::v1::ReduceMax>(
            {ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape()), ov::pass::pattern::wrap_type<ov::op::v0::Constant>()},
            ov::pass::pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_pooling<ov::op::v1::ReduceMax>());
}
ov::pass::ConvertReduceSumToPooling::ConvertReduceSumToPooling() {
    MATCHER_SCOPE(ConvertReduceSumToPooling);
    auto m = std::make_shared<ov::pass::pattern::Matcher>(
        ov::pass::pattern::wrap_type<ov::op::v1::ReduceSum>(
            {ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape()), ov::pass::pattern::wrap_type<ov::op::v0::Constant>()},
            ov::pass::pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_pooling<ov::op::v1::ReduceSum>());
}
