// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_negative.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using ov::pass::pattern::Matcher;

namespace v0 = ov::op::v0;
ov::pass::ConvertNegative::ConvertNegative() {
    MATCHER_SCOPE(ConvertNegative);
    auto neg = ov::pass::pattern::wrap_type<v0::Negative>();

    matcher_pass_callback callback = [](Matcher& m) {
        auto neg = ov::as_type_ptr<v0::Negative>(m.get_match_root());
        if (!neg) {
            return false;
        }

        auto mul = std::make_shared<ov::op::v1::Multiply>(neg->input(0).get_source_output(),
                                                          v0::Constant::create(neg->get_element_type(), Shape{}, {-1}));
        mul->set_friendly_name(neg->get_friendly_name());
        ov::copy_runtime_info(neg, mul);
        ov::replace_node(neg, mul);
        return true;
    };

    auto m = std::make_shared<Matcher>(neg, matcher_name);
    this->register_matcher(m, callback);
}
