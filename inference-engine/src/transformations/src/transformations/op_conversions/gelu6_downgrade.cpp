// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/gelu6_downgrade.hpp"

#include <memory>

#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::Gelu6Downgrade, "Gelu6Downgrade", 0);

ngraph::pass::Gelu6Downgrade::Gelu6Downgrade() {
    MATCHER_SCOPE(Gelu6Downgrade);
    auto gelu = ngraph::pattern::wrap_type<opset6::Gelu>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto gelu_node = std::dynamic_pointer_cast<ngraph::opset6::Gelu>(pattern_to_output.at(gelu).get_node_shared_ptr());

        if (gelu_node == nullptr || transformation_callback(gelu_node)) {
            return false;
        }

        auto new_gelu_node = std::make_shared<ngraph::opset2::Gelu>();
        new_gelu_node->set_friendly_name(gelu_node->get_friendly_name());
        ngraph::copy_runtime_info(gelu_node, new_gelu_node);
        ngraph::replace_node(gelu_node, new_gelu_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gelu, matcher_name);
    register_matcher(m, callback);
}
