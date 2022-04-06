// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class ConvertProposalToLegacyMatcher;
class ConvertProposal4ToLegacyMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertProposal4ToLegacyMatcher: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertProposal4ToLegacyMatcher", "0");
    ConvertProposal4ToLegacyMatcher();
};

class ngraph::pass::ConvertProposalToLegacyMatcher: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertProposalToLegacyMatcher", "0");
    ConvertProposalToLegacyMatcher();
};
