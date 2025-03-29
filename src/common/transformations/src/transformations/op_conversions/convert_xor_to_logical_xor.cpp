// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_xor_to_logical_xor.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/xor.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ConvertXorToLogicalXor::ConvertXorToLogicalXor() {
    MATCHER_SCOPE(ConvertXorToLogicalXor);

    auto xor_v1 = pattern::wrap_type<ov::op::v0::Xor>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto xor_v1_node = ov::as_type_ptr<ov::op::v0::Xor>(m.get_match_root());
        if (!xor_v1_node)
            return false;

        const auto& autobroad = xor_v1_node->get_autob();

        auto logical_xor_v10 = std::make_shared<ov::op::v1::LogicalXor>(xor_v1_node->input_value(0),
                                                                        xor_v1_node->input_value(1),
                                                                        autobroad);
        logical_xor_v10->set_friendly_name(xor_v1_node->get_friendly_name());
        ov::copy_runtime_info(xor_v1_node, logical_xor_v10);
        ov::replace_node(xor_v1_node, logical_xor_v10);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(xor_v1, matcher_name);
    register_matcher(m, callback);
}
