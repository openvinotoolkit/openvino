// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/smart_reshape/reshape_to_1D.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

ngraph::pass::ReshapeTo1D::ReshapeTo1D() {
    auto reshape_label = ngraph::pattern::wrap_type<opset5::Reshape>({pattern::any_input(), ngraph::pattern::wrap_type<opset5::Constant>()});

    matcher_pass_callback callback = [=](pattern::Matcher &m) -> bool {
        const auto &pattern_to_output = m.get_pattern_value_map();

        const auto & reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();
        const auto & output_pshape =  reshape->get_output_partial_shape(0);
        if (output_pshape.rank().is_static() && output_pshape.rank().get_length() == 1) {
            reshape->input(1).replace_source_output(ngraph::opset5::Constant::create(ngraph::element::i64, {1}, {-1}));
            return true;
        }
        return false;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_label, "ReshapeTo1D");
    register_matcher(m, callback);
}
