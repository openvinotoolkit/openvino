// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/smart_reshape/proposal_scales_stridedslice.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

bool crop_scales_for_proposal(const ngraph::pattern::PatternValueMap & pattern_to_output,
                              std::shared_ptr<ngraph::Node> parameter_label, std::shared_ptr<ngraph::Node> proposal_label) {
    const auto & parameter = pattern_to_output.at(parameter_label);
    const auto & proposal = pattern_to_output.at(proposal_label).get_node_shared_ptr();

    auto cropped_scales = std::make_shared<ngraph::opset5::StridedSlice>(
            proposal->input_value(2),
            ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}),
            ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {parameter.get_partial_shape()[1].get_length()}),
            ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1}),
            std::vector<int64_t>{0}, std::vector<int64_t>{0});

    proposal->input(2).replace_source_output(cropped_scales->output(0));
    return true;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::Proposal1Scales, "Proposal1Scales", 0);

ngraph::pass::Proposal1Scales::Proposal1Scales() {
    auto parameter_label = ngraph::pattern::wrap_type<opset5::Parameter>([](const Output<Node> &output) {
        const auto & shape = output.get_partial_shape();
        return shape.rank().is_static() && shape.rank().get_length() == 2 && shape[1].is_static() && (shape[1].get_length() == 3 || shape[1].get_length() == 4);
    });
    auto reshape_label = ngraph::pattern::wrap_type<opset5::Reshape>({parameter_label, ngraph::pattern::wrap_type<opset5::Constant>()},
         [](const Output<Node> &output) { return output.get_partial_shape().rank().is_static() && output.get_partial_shape().rank().get_length() == 1; });
    auto proposal_label = ngraph::pattern::wrap_type<opset1::Proposal>({pattern::any_input(), pattern::any_input(), reshape_label});

    matcher_pass_callback callback = [parameter_label, proposal_label](pattern::Matcher &m) -> bool {
        return crop_scales_for_proposal(m.get_pattern_value_map(), parameter_label, proposal_label);
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(proposal_label, "Proposal1Scales");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::Proposal4Scales, "Proposal4Scales", 0);

ngraph::pass::Proposal4Scales::Proposal4Scales() {
    auto parameter_label = ngraph::pattern::wrap_type<opset5::Parameter>([](const Output<Node> &output) {
        const auto & shape = output.get_partial_shape();
        return shape.rank().is_static() && shape.rank().get_length() == 2 && shape[1].is_static() && (shape[1].get_length() == 3 || shape[1].get_length() == 4);
    });
    auto reshape_label = ngraph::pattern::wrap_type<opset5::Reshape>({parameter_label, ngraph::pattern::wrap_type<opset5::Constant>()},
         [](const Output<Node> &output) { return output.get_partial_shape().rank().is_static() && output.get_partial_shape().rank().get_length() == 1; });
    auto proposal_label = ngraph::pattern::wrap_type<opset4::Proposal>({pattern::any_input(), pattern::any_input(), reshape_label});

    matcher_pass_callback callback = [parameter_label, proposal_label](pattern::Matcher &m) -> bool {
        return crop_scales_for_proposal(m.get_pattern_value_map(), parameter_label, proposal_label);
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(proposal_label, "Proposal4Scales");
    register_matcher(m, callback);
}
