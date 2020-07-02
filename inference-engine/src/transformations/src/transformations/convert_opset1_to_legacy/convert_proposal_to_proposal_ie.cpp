// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_proposal_to_proposal_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

#include <ngraph_ops/proposal_ie.hpp>
#include <ngraph/rt_info.hpp>

ngraph::pass::ConvertProposalToLegacyMatcher::ConvertProposalToLegacyMatcher() {
    ngraph::handler_callback callback = [](const std::shared_ptr<Node>& node) -> bool {
        auto proposal = std::dynamic_pointer_cast<ngraph::opset1::Proposal>(node);
        if (!proposal) {
            return false;
        }

        Output<Node> last;

        ngraph::NodeVector ops_to_replace, new_ops;
        ops_to_replace.push_back(proposal);

        if (auto reshape = std::dynamic_pointer_cast<opset1::Reshape>(proposal->input_value(2).get_node_shared_ptr())) {
            auto input_shape = reshape->get_input_shape(0);
            if (input_shape.size() == 2) {
                last = reshape->input_value(0);
                ops_to_replace.push_back(reshape);
            }
        }

        if (!last.get_node_shared_ptr()) {
            std::vector<int64_t> dims{1, -1};
            auto const_shape = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{2}, dims);
            last = std::make_shared<ngraph::opset1::Reshape>(proposal->input_value(2), const_shape, true);
            new_ops.push_back(last.get_node_shared_ptr());
        }

        auto proposal_ie = std::make_shared<ngraph::op::ProposalIE> (proposal->input_value(0),
                                                                     proposal->input_value(1),
                                                                     last,
                                                                     proposal->get_attrs());
        new_ops.push_back(proposal_ie);

        proposal_ie->set_friendly_name(proposal->get_friendly_name());
        ngraph::copy_runtime_info(ops_to_replace, new_ops);
        ngraph::replace_node(proposal, proposal_ie);
        return true;
    };

    this->register_matcher(callback);
}