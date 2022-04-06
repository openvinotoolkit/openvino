// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/smart_reshape/proposal_scales_stridedslice.hpp>

#include "itt.hpp"

namespace {

bool crop_scales_for_proposal(const ngraph::pattern::PatternValueMap& pattern_to_output,
                              const std::shared_ptr<ngraph::Node>& parameter_label,
                              const std::shared_ptr<ngraph::Node>& proposal_label) {
    const auto& parameter = pattern_to_output.at(parameter_label);
    const auto& proposal = pattern_to_output.at(proposal_label).get_node_shared_ptr();

    auto cropped_scales = std::make_shared<ngraph::opset5::StridedSlice>(
        proposal->input_value(2),
        ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}),
        ngraph::opset5::Constant::create(ngraph::element::i64,
                                         ngraph::Shape{1},
                                         {parameter.get_partial_shape()[1].get_length()}),
        ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1}),
        std::vector<int64_t>{0},
        std::vector<int64_t>{0});

    proposal->input(2).replace_source_output(cropped_scales->output(0));
    return true;
}

}  // namespace

ngraph::pass::Proposal1Scales::Proposal1Scales() {
    // TODO: enable conditional compile
    // MATCHER_SCOPE(Proposal1Scales);
    auto parameter_label = ngraph::pattern::wrap_type<opset5::Parameter>([](const Output<Node>& output) {
        const auto& shape = output.get_partial_shape();
        return shape.rank().is_static() && shape.rank().get_length() == 2 && shape[1].is_static() &&
               (shape[1].get_length() == 3 || shape[1].get_length() == 4);
    });
    auto convert_label = ngraph::pattern::wrap_type<opset5::Convert>({parameter_label});
    auto param_or_convert =
        std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{parameter_label, convert_label});
    auto reshape_label = ngraph::pattern::wrap_type<opset5::Reshape>(
        {param_or_convert, ngraph::pattern::wrap_type<opset5::Constant>()},
        [](const Output<Node>& output) {
            return output.get_partial_shape().rank().is_static() && output.get_partial_shape().rank().get_length() == 1;
        });
    auto proposal_label =
        ngraph::pattern::wrap_type<opset1::Proposal>({pattern::any_input(), pattern::any_input(), reshape_label});

    matcher_pass_callback callback = [parameter_label, proposal_label](pattern::Matcher& m) -> bool {
        return crop_scales_for_proposal(m.get_pattern_value_map(), parameter_label, proposal_label);
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(proposal_label /*, matcher_name */);
    register_matcher(m, callback);
}

ngraph::pass::Proposal4Scales::Proposal4Scales() {
    // TODO: enable conditional compile
    // MATCHER_SCOPE(Proposal4Scales);
    auto parameter_label = ngraph::pattern::wrap_type<opset5::Parameter>([](const Output<Node>& output) {
        const auto& shape = output.get_partial_shape();
        return shape.rank().is_static() && shape.rank().get_length() == 2 && shape[1].is_static() &&
               (shape[1].get_length() == 3 || shape[1].get_length() == 4);
    });
    auto convert_label = ngraph::pattern::wrap_type<opset5::Convert>({parameter_label});
    auto param_or_convert =
        std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{parameter_label, convert_label});
    auto reshape_label = ngraph::pattern::wrap_type<opset5::Reshape>(
        {param_or_convert, ngraph::pattern::wrap_type<opset5::Constant>()},
        [](const Output<Node>& output) {
            return output.get_partial_shape().rank().is_static() && output.get_partial_shape().rank().get_length() == 1;
        });
    auto proposal_label =
        ngraph::pattern::wrap_type<opset4::Proposal>({pattern::any_input(), pattern::any_input(), reshape_label});

    matcher_pass_callback callback = [parameter_label, proposal_label](pattern::Matcher& m) -> bool {
        return crop_scales_for_proposal(m.get_pattern_value_map(), parameter_label, proposal_label);
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(proposal_label /*, matcher_name */);
    register_matcher(m, callback);
}
