// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

class CommonOptimizations : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    CommonOptimizations();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
