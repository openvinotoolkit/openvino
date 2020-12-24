// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_mvn1.hpp"

#include <ngraph/rt_info.hpp>

#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertMVN1, "ConvertMVN1", 0);

ngraph::pass::ConvertMVN1::ConvertMVN1() {
    auto mvn = pattern::wrap_type<opset2::MVN>(); // MVN was missing in opset1

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto mvn_node = std::dynamic_pointer_cast<opset2::MVN>(m.get_match_root());
        if (!mvn_node) {
            return false;
        }

        const auto input = mvn_node->input_value(0);

        // MVN-1 support only 4D input tensors
        if (input.get_partial_shape().rank().is_static() && input.get_partial_shape().rank().get_length() == 4) {
            std::shared_ptr<ngraph::opset6::Constant> axes;
            if (mvn_node->get_across_channels()) {
                axes = opset6::Constant::create(ngraph::element::i64, { 3 }, { 1, 2, 3 });
            } else {
                axes = opset6::Constant::create(ngraph::element::i64, { 2 }, { 2, 3 });
            }
            auto mvn6_node = std::make_shared<ngraph::opset6::MVN>(input,
                axes,
                mvn_node->get_normalize_variance(),
                mvn_node->get_eps(),
                ngraph::op::MVNEpsMode::OUTSIDE_SQRT);

            mvn6_node->set_friendly_name(m.get_match_root()->get_friendly_name());
            ngraph::copy_runtime_info(mvn_node, { axes, mvn6_node });
            ngraph::replace_node(m.get_match_root(), mvn6_node);
            return true;
        } else {
            return false;
        }
    };

    auto m = std::make_shared<pattern::Matcher>(mvn, "ConvertMVN1");
    register_matcher(m, callback);
}
