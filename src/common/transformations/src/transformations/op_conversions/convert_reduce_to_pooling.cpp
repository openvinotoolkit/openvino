// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"

#include "itt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"


using ov::pass::pattern::any_input;
using ov::pass::pattern::wrap_type;
using ov::pass::pattern::Matcher;
using ov::pass::pattern::has_static_shape;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
ov::pass::ConvertReduceMeanToPooling::ConvertReduceMeanToPooling() {
    MATCHER_SCOPE(ConvertReduceMeanToPooling);
    auto m = std::make_shared<Matcher>(
        wrap_type<v1::ReduceMean>(
            {any_input(has_static_shape()),
             wrap_type<v0::Constant>()},
            has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_pooling<v1::ReduceMean>());
}
ov::pass::ConvertReduceMaxToPooling::ConvertReduceMaxToPooling() {
    MATCHER_SCOPE(ConvertReduceMaxToPooling);
    auto m = std::make_shared<Matcher>(
        wrap_type<v1::ReduceMax>(
            {any_input(has_static_shape()),
             wrap_type<v0::Constant>()},
            has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_pooling<v1::ReduceMax>());
}
ov::pass::ConvertReduceSumToPooling::ConvertReduceSumToPooling() {
    MATCHER_SCOPE(ConvertReduceSumToPooling);
    auto m = std::make_shared<Matcher>(
        wrap_type<v1::ReduceSum>(
            {any_input(has_static_shape()),
             wrap_type<v0::Constant>()},
            has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_pooling<v1::ReduceSum>());
}
