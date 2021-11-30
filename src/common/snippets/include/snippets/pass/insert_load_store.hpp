// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface InsertLoad
 * @brief Inserts explicit load instruction after each parameter.
 * The pass is used to convert function to a canonical form for code generation
 * @ingroup snippets
 */
class TRANSFORMATIONS_API InsertLoad: public ngraph::pass::MatcherPass {
public:
    InsertLoad();
};

/**
 * @interface InsertStore
 * @brief Inserts explicit store instruction before each result.
 * The pass is used to convert function to a canonical form for code generation
 * @ingroup snippets
 */
class TRANSFORMATIONS_API InsertStore: public ngraph::pass::MatcherPass {
public:
    InsertStore();
};


}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
