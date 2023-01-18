// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface LoopFusion
 * @brief Fuse Loops into one Loop if their semantics allow it
 * @ingroup snippets
 */
class LoopFusion: public ngraph::pass::MatcherPass {
public:
    LoopFusion();

private:
    bool Merge(const std::shared_ptr<op::LoopBegin>& buffer);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
