// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_proposal_to_proposal_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

#include <ngraph_ops/proposal_ie.hpp>

void ngraph::pass::ConvertProposalToProposalIE::convert_proposal() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto input_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{3});

    ngraph::op::ProposalAttrs attr = {};

    auto proposal = std::make_shared<ngraph::opset1::Proposal>(input_0, input_1, input_2, attr);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto proposal = std::dynamic_pointer_cast<ngraph::opset1::Proposal> (m.get_match_root());

        if (!proposal) {
            return false;
        }

        Output<Node> last;

        if (auto reshape = std::dynamic_pointer_cast<opset1::Reshape>(proposal->input_value(2).get_node_shared_ptr())) {
            auto input_shape = reshape->get_input_shape(0);
            if (input_shape.size() == 2) {
                last = reshape->input_value(0);
            }
        }

        if (!last.get_node_shared_ptr()) {
            std::vector<int64_t> dims{1, -1};
            auto const_shape = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{2}, dims);
            last = std::make_shared<ngraph::opset1::Reshape>(proposal->input_value(2), const_shape, true);
        }

        auto proposal_ie = std::make_shared<ngraph::op::ProposalIE> (proposal->input_value(0),
                                                                     proposal->input_value(1),
                                                                     last,
                                                                     proposal->get_attrs());

        proposal_ie->set_friendly_name(proposal->get_friendly_name());
        ngraph::replace_node(m.get_match_root(), proposal_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(proposal, "CPUFusion.ConvertProposalToProposalIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}