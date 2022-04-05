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
 * @interface LoadMoveBroadcastToBroadcastLoad
 * @brief Fuses consecutive Load and MoveBroadcast into a single load insctruction.
 * The pass is used to convert model to a canonical form for code generation
 * @ingroup snippets
 */
class LoadMoveBroadcastToBroadcastLoad: public ngraph::pass::MatcherPass {
public:
    LoadMoveBroadcastToBroadcastLoad();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
