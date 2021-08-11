// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/gelu7_downgrade.hpp"

#include <memory>

#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ov::pass::Gelu7Downgrade, "Gelu7Downgrade", 0);

ov::pass::Gelu7Downgrade::Gelu7Downgrade() {
    MATCHER_SCOPE(Gelu7Downgrade);
    auto gelu = ov::pattern::wrap_type<opset7::Gelu>();

    ov::matcher_pass_callback callback = [=](ov::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto gelu_node = std::dynamic_pointer_cast<ov::opset7::Gelu>(pattern_to_output.at(gelu).get_node_shared_ptr());

        if (gelu_node == nullptr || transformation_callback(gelu_node)) {
            return false;
        }

        auto new_gelu_node = std::make_shared<ov::opset2::Gelu>(gelu_node->input_value(0));
        new_gelu_node->set_friendly_name(gelu_node->get_friendly_name());
        ov::copy_runtime_info(gelu_node, new_gelu_node);
        ov::replace_node(gelu_node, new_gelu_node);
        return true;
    };

    auto m = std::make_shared<ov::pattern::Matcher>(gelu, matcher_name);
    register_matcher(m, callback);
}
