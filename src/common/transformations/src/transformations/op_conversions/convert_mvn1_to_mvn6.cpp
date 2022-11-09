// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_mvn1_to_mvn6.hpp"

#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <numeric>

#include "itt.hpp"

ngraph::pass::ConvertMVN1ToMVN6::ConvertMVN1ToMVN6() {
    MATCHER_SCOPE(ConvertMVN1ToMVN6);
    auto mvn = pattern::wrap_type<ngraph::opset2::MVN>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto mvn_node = std::dynamic_pointer_cast<ngraph::opset2::MVN>(m.get_match_root());
        if (!mvn_node) {
            return false;
        }

        const auto input = mvn_node->input_value(0);
        auto input_rank = input.get_partial_shape().rank();
        if (!input_rank.is_static()) {
            return false;
        }
        int64_t start_axis = 1 + static_cast<int64_t>(!mvn_node->get_across_channels());
        if (input_rank.get_length() <= start_axis) {
            return false;
        }
        std::vector<int64_t> axes_v(input_rank.get_length() - start_axis);
        std::iota(axes_v.begin(), axes_v.end(), start_axis);
        auto axes = opset6::Constant::create(ngraph::element::i64, {axes_v.size()}, axes_v);
        auto mvn6_node = std::make_shared<ngraph::opset6::MVN>(input,
                                                               axes,
                                                               mvn_node->get_normalize_variance(),
                                                               static_cast<float>(mvn_node->get_eps()),
                                                               ngraph::op::MVNEpsMode::OUTSIDE_SQRT);

        mvn6_node->set_friendly_name(mvn_node->get_friendly_name());
        ngraph::copy_runtime_info(mvn_node, mvn6_node);
        ngraph::replace_node(mvn_node, mvn6_node);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(mvn, matcher_name);
    register_matcher(m, callback);
}
