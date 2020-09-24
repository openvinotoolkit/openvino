// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertProposalToLegacyMatcher;
class TRANSFORMATIONS_API ConvertProposal4ToLegacyMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertProposal4ToLegacyMatcher: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertProposal4ToLegacyMatcher();
};

class ngraph::pass::ConvertProposalToLegacyMatcher: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertProposalToLegacyMatcher();
};
