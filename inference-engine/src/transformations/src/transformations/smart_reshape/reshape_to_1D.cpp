// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include <transformations/smart_reshape/reshape_to_1D.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

NGRAPH_RTTI_DEFINITION(ov::pass::ReshapeTo1D, "ReshapeTo1D", 0);

ov::pass::ReshapeTo1D::ReshapeTo1D() {
    MATCHER_SCOPE(ReshapeTo1D);
    auto reshape_label = ov::pattern::wrap_type<opset5::Reshape>({pattern::any_input(), ov::pattern::wrap_type<opset5::Constant>()},
             [](const Output<Node> & output) { return output.get_partial_shape().rank().is_static() && output.get_partial_shape().rank().get_length() == 1; });

    matcher_pass_callback callback = [](pattern::Matcher &m) -> bool {
        m.get_match_root()->input(1).replace_source_output(ov::opset5::Constant::create(ov::element::i64, {1}, {-1}));
        return true;
    };
    auto m = std::make_shared<ov::pattern::Matcher>(reshape_label, matcher_name);
    register_matcher(m, callback);
}
