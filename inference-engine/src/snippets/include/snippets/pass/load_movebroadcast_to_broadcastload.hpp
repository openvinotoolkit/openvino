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
 * @interface LoadMoveBroadcastToBroadcastLoad
 * @brief Fuses consecutive Load and MoveBroadcast into a single load insctruction.
 * The pass is used to convert function to a canonical form for code generation
 * @ingroup snippets
 */
class TRANSFORMATIONS_API LoadMoveBroadcastToBroadcastLoad: public ngraph::pass::MatcherPass {
public:
    LoadMoveBroadcastToBroadcastLoad();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
