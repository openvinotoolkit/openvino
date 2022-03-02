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
 * @interface SetScalarCountForLoad
 * @brief Set count `1` for Load to represent as ScalarLoad
 * The pass is used to range element type of function in a canonical form vector to scalar.
 * Used for tail generation
 * @ingroup snippets
 */
class SetScalarCountForLoad: public ngraph::pass::MatcherPass {
public:
    SetScalarCountForLoad();
};

/**
 * @interface SetScalarCountForStore
 * @brief Set count `1` for Store to represent as ScalarStore
 * The pass is used to range element type of function in a canonical form vector to scalar.
 * Used for tail generation
 * @ingroup snippets
 */
class SetScalarCountForStore: public ngraph::pass::MatcherPass {
public:
    SetScalarCountForStore();
};

} // namespace pass
} // namespace snippets
} // namespace ngraph
