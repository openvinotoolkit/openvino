// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/convert_proposal_to_proposal_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <legacy/ngraph_ops/proposal_ie.hpp>
#include <ngraph/rt_info.hpp>

namespace {

bool convert_to_proposal_ie(std::shared_ptr<ngraph::op::v0::Proposal> proposal, bool infer_probs = false) {
    ngraph::Output<ngraph::Node> last; // 2D tensor of size [1, 3-4] with im_info will be retrieved from this node
    ngraph::NodeVector ops_to_replace, new_ops;
    ops_to_replace.push_back(proposal);

    if (auto reshape = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(proposal->input_value(2).get_node_shared_ptr())) {
        const ngraph::PartialShape& im_info_shape = reshape->get_input_partial_shape(0);
        if (im_info_shape != ngraph::Shape({1, 3}) && im_info_shape != ngraph::Shape({1, 4})) {
            return false;
        }
        last = reshape->input_value(0);
        ops_to_replace.push_back(reshape);
    } else {
        auto const_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, -1});
        last = std::make_shared<ngraph::opset1::Reshape>(proposal->input_value(2), const_shape, true);
        new_ops.push_back(last.get_node_shared_ptr());
    }

    auto ie_attrs = proposal->get_attrs();
    ie_attrs.infer_probs = infer_probs;
    auto proposal_ie = std::make_shared<ngraph::op::ProposalIE>(proposal->input_value(0),
                                                                proposal->input_value(1),
                                                                last,
                                                                ie_attrs);
    new_ops.push_back(proposal_ie);

    proposal_ie->set_friendly_name(proposal->get_friendly_name());
    ngraph::copy_runtime_info(ops_to_replace, new_ops);
    ngraph::replace_node(proposal, proposal_ie);

    return true;
}

} // namespace

ngraph::pass::ConvertProposalToLegacyMatcher::ConvertProposalToLegacyMatcher() {
    auto proposal = ngraph::pattern::wrap_type<ngraph::opset1::Proposal>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto proposal = std::dynamic_pointer_cast<ngraph::opset1::Proposal>(m.get_match_root());

        if (!proposal) {
            return false;
        }
        convert_to_proposal_ie(proposal);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(proposal, "ConvertProposalToProposalIE");
    this->register_matcher(m, callback);
}

ngraph::pass::ConvertProposal4ToLegacyMatcher::ConvertProposal4ToLegacyMatcher() {
    auto proposal = ngraph::pattern::wrap_type<ngraph::opset4::Proposal>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto proposal = std::dynamic_pointer_cast<ngraph::opset4::Proposal>(m.get_match_root());

        if (!proposal) {
            return false;
        }
        convert_to_proposal_ie(proposal, true);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(proposal, "ConvertProposal4ToProposalIE");
    this->register_matcher(m, callback);
}
