// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface TokenizeMHASnippets
 * @brief The pass tokenizes MHA-pattern into Subgraph
 *        TODO: Write pattern
 * @ingroup snippets
 */
class TokenizeMHASnippets: public ngraph::pass::MatcherPass {
public:
    TokenizeMHASnippets();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
