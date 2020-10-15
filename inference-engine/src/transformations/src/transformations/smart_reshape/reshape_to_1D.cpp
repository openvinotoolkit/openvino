// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/smart_reshape/reshape_to_1D.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ReshapeTo1D, "ReshapeTo1D", 0);

ngraph::pass::ReshapeTo1D::ReshapeTo1D() {
    auto reshape_label = ngraph::pattern::wrap_type<opset5::Reshape>({pattern::any_input(), ngraph::pattern::wrap_type<opset5::Constant>()},
             [](const Output<Node> & output) { return output.get_partial_shape().rank().is_static() && output.get_partial_shape().rank().get_length() == 1; });

    matcher_pass_callback callback = [](pattern::Matcher &m) -> bool {
        m.get_match_root()->input(1).replace_source_output(ngraph::opset5::Constant::create(ngraph::element::i64, {1}, {-1}));
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_label, "ReshapeTo1D");
    register_matcher(m, callback);
}
