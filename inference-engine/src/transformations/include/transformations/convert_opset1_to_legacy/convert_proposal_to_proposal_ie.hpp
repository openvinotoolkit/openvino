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

class TRANSFORMATIONS_API ConvertProposalToProposalIEMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertProposalToProposalIEMatcher {
public:
    void register_matcher(std::shared_ptr<ngraph::pass::GraphRewrite> t);
};
