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
 * @interface InsertLoad
 * @brief Inserts explicit load instruction after each parameter.
 * The pass is used to convert model to a canonical form for code generation
 * @ingroup snippets
 */
class InsertLoad: public ngraph::pass::MatcherPass {
public:
    InsertLoad();
};

/**
 * @interface InsertStore
 * @brief Inserts explicit store instruction before each result.
 * The pass is used to convert model to a canonical form for code generation
 * @ingroup snippets
 */
class InsertStore: public ngraph::pass::MatcherPass {
public:
    InsertStore();
};


}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
