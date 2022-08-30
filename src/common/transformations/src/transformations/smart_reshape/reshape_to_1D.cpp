// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/smart_reshape/reshape_to_1D.hpp>

#include "itt.hpp"

ngraph::pass::ReshapeTo1D::ReshapeTo1D() {
    // TODO: enable conditional compile
    // MATCHER_SCOPE(ReshapeTo1D);
    auto reshape_label = ngraph::pattern::wrap_type<opset5::Reshape>(
        {pattern::any_input(), ngraph::pattern::wrap_type<opset5::Constant>()},
        [](const Output<Node>& output) {
            return output.get_partial_shape().rank().is_static() && output.get_partial_shape().rank().get_length() == 1;
        });

    matcher_pass_callback callback = [](pattern::Matcher& m) -> bool {
        m.get_match_root()->input(1).replace_source_output(
            ngraph::opset5::Constant::create(ngraph::element::i64, {1}, {-1}));
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_label /*, matcher_name*/);
    register_matcher(m, callback);
}
