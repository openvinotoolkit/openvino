// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/proposal_ie.hpp>

#include "ngraph/op/experimental/dyn_slice.hpp"
#include "ngraph/op/experimental/layers/proposal.hpp"
#include "ngraph/op/constant.hpp"

namespace ngraph {
namespace pass {

class ConvertProposalToProposalIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertProposalToProposalIE: public ngraph::pass::GraphRewrite {
public:
    ConvertProposalToProposalIE() : GraphRewrite() {
        convert_proposal();
    }

private:
    void convert_proposal();
};

void ngraph::pass::ConvertProposalToProposalIE::convert_proposal() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto input_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{3});

    ngraph::op::ProposalAttrs attr = {};

    auto proposal = std::make_shared<ngraph::op::Proposal>(input_0, input_1, input_2, attr);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto proposal = std::dynamic_pointer_cast<ngraph::op::Proposal> (m.get_match_root());

        if (!proposal) {
            return false;
        }

        std::vector<int64_t> dims{1, -1};
        auto const_shape = std::make_shared<ngraph::op::Constant> (element::i64, Shape{2}, dims);
        auto reshape = std::make_shared<ngraph::op::DynReshape> (proposal->get_argument(2), const_shape);

        auto proposal_ie = std::make_shared<ngraph::op::ProposalIE> (proposal->get_argument(0),
                                                                     proposal->get_argument(1),
                                                                     reshape,
                                                                     proposal->get_attrs());

        proposal_ie->set_friendly_name(proposal->get_friendly_name());
        ngraph::replace_node(m.get_match_root(), proposal_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(proposal, "CPUFusion.ConvertProposalToProposalIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
