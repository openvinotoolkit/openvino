// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/convert_swish_to_swish_ie.hpp"

#include <memory>

#include <ngraph/opsets/opset4.hpp>

#include <legacy/ngraph_ops/swish_ie.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertSwishToSwishIEMatcher, "ConvertSwishToSwishIEMatcher", 0);

ngraph::pass::ConvertSwishToSwishIEMatcher::ConvertSwishToSwishIEMatcher() {
    auto swish = ngraph::pattern::wrap_type<ngraph::opset4::Swish>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto swish = std::dynamic_pointer_cast<ngraph::opset4::Swish> (m.get_match_root());
        if (!swish) {
            return false;
        }
        float beta_value = 1.0;
        if (swish->input_values().size() == 2) {
            auto beta_node = swish->input_value(1).get_node_shared_ptr();
            auto beta_const = std::dynamic_pointer_cast<ngraph::opset4::Constant>(beta_node);

            if (!beta_const) {
                return false;
            }
            if (!ngraph::op::util::get_single_value(beta_const, beta_value)) {
                return false;
            }
        }

        auto swish_ie = std::make_shared<ngraph::op::SwishIE>(swish->input(0).get_source_output(), beta_value);
        swish_ie->set_friendly_name(swish->get_friendly_name());
        ngraph::copy_runtime_info(swish, swish_ie);
        ngraph::replace_node(swish, swish_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(swish, "ConvertSwishToSwishIE");
    this->register_matcher(m, callback);
}