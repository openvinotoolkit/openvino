// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class ConvertProposalToLegacyMatcher;
class ConvertProposal4ToLegacyMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertProposal4ToLegacyMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertProposal4ToLegacyMatcher", "0");
    ConvertProposal4ToLegacyMatcher();
};

class ngraph::pass::ConvertProposalToLegacyMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertProposalToLegacyMatcher", "0");
    ConvertProposalToLegacyMatcher();
};
