// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"

#include "itt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace ov::pass {

ConvertReduceMeanToPooling::ConvertReduceMeanToPooling() {
    MATCHER_SCOPE(ConvertReduceMeanToPooling);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<v1::ReduceMean>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_pooling<v1::ReduceMean>());
}

ConvertReduceMaxToPooling::ConvertReduceMaxToPooling() {
    MATCHER_SCOPE(ConvertReduceMaxToPooling);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<v1::ReduceMax>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_pooling<v1::ReduceMax>());
}

ConvertReduceSumToPooling::ConvertReduceSumToPooling() {
    MATCHER_SCOPE(ConvertReduceSumToPooling);
    auto m = std::make_shared<pattern::Matcher>(
        pattern::wrap_type<v1::ReduceSum>(
            {pattern::any_input(pattern::has_static_shape()), pattern::wrap_type<v0::Constant>()},
            pattern::has_static_shape()),
        matcher_name);
    register_matcher(m, convert_reduce_to_pooling<v1::ReduceSum>());
}

}  // namespace ov::pass
