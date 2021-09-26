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
* @interface MulAddToFMA
* @brief Replaces mul and add with FMA node
* @ingroup snippets
*/
class MulAddToFMA : public ngraph::pass::MatcherPass {
public:
    MulAddToFMA();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
