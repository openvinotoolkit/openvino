// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/gelu7_downgrade.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::Gelu7Downgrade::Gelu7Downgrade() {
    MATCHER_SCOPE(Gelu7Downgrade);
    auto gelu = ov::pass::pattern::wrap_type<ov::op::v7::Gelu>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto gelu_node = ov::as_type_ptr<ov::op::v7::Gelu>(pattern_to_output.at(gelu).get_node_shared_ptr());

        if (gelu_node == nullptr || transformation_callback(gelu_node)) {
            return false;
        }

        auto new_gelu_node = std::make_shared<ov::op::v0::Gelu>(gelu_node->input_value(0));
        new_gelu_node->set_friendly_name(gelu_node->get_friendly_name());
        ov::copy_runtime_info(gelu_node, new_gelu_node);
        ov::replace_node(gelu_node, new_gelu_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gelu, matcher_name);
    register_matcher(m, callback);
}
