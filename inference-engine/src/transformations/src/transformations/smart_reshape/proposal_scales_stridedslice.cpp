// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/smart_reshape/proposal_scales_stridedslice.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

ngraph::pass::ProposalScales::ProposalScales() {
    auto parameter_label = ngraph::pattern::wrap_type<opset5::Parameter>();
    auto reshape_label = ngraph::pattern::wrap_type<opset5::Reshape>({parameter_label, ngraph::pattern::wrap_type<opset5::Constant>()});
    auto proposal_label = ngraph::pattern::wrap_type<opset5::Proposal>({pattern::any_input(), pattern::any_input(), reshape_label});

    matcher_pass_callback callback = [=](pattern::Matcher &m) -> bool {
        const auto &pattern_to_output = m.get_pattern_value_map();

        const auto & parameter = pattern_to_output.at(parameter_label).get_node_shared_ptr();
        const auto & reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();
        const auto & proposal = pattern_to_output.at(proposal_label).get_node_shared_ptr();

        const auto & parameter_pshape = parameter->get_output_partial_shape(0);

        if (parameter_pshape.rank().is_dynamic() || parameter_pshape.rank().get_length() != 2)
            return false;
        if (reshape->get_output_partial_shape(0).rank().is_dynamic() || reshape->get_output_partial_shape(0).rank() != 1)
            return false;
        if (parameter_pshape[1].is_dynamic() || (parameter_pshape[1].get_length() != 3 && parameter_pshape[1].get_length() != 4))
            return false;

        auto begin  = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto end    = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {parameter_pshape[1].get_length()});
        auto stride = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto ss_data = std::make_shared<ngraph::opset5::StridedSlice>(
                proposal->input_value(2), begin, end, stride, std::vector<int64_t>{0}, std::vector<int64_t>{0});

        proposal->input(2).replace_source_output(ss_data->output(0));
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(proposal_label, "ProposalScales");
    register_matcher(m, callback);
}
