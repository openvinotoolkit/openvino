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
 * @interface ReplaceLoadsWithScalarLoads
 * @brief Replases vector loads with scalar versions.
 * The pass is used to cange alement type of function in a canonical form vector to scalar.
 * Used for tail generation
 * @ingroup snippets
 */
class TRANSFORMATIONS_API ReplaceLoadsWithScalarLoads: public ngraph::pass::MatcherPass {
public:
    ReplaceLoadsWithScalarLoads();
};

/**
 * @interface ReplaceStoresWithScalarStores
 * @brief Replases vector stores with scalar versions.
 * The pass is used to cange alement type of function in a canonical form vector to scalar.
 * Used for tail generation
 * @ingroup snippets
 */
class TRANSFORMATIONS_API ReplaceStoresWithScalarStores: public ngraph::pass::MatcherPass {
public:
    ReplaceStoresWithScalarStores();
};

} // namespace pass
} // namespace snippets
} // namespace ngraph
