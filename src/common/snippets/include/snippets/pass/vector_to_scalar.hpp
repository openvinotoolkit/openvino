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
 * @interface ReplaceLoadsWithScalarLoads
 * @brief Replaces vector loads with scalar versions.
 * The pass is used to cange element type of function in a canonical form vector to scalar.
 * Used for tail generation
 * @ingroup snippets
 */
class ReplaceLoadsWithScalarLoads: public ngraph::pass::MatcherPass {
public:
    ReplaceLoadsWithScalarLoads();
};

/**
 * @interface ReplaceStoresWithScalarStores
 * @brief Replaces vector stores with scalar versions.
 * The pass is used to cange element type of model in a canonical form vector to scalar.
 * Used for tail generation
 * @ingroup snippets
 */
class ReplaceStoresWithScalarStores: public ngraph::pass::MatcherPass {
public:
    ReplaceStoresWithScalarStores();
};

} // namespace pass
} // namespace snippets
} // namespace ngraph
