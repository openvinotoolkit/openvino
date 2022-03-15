// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

    class TRANSFORMATIONS_API EliminateUselessMul;
    class TRANSFORMATIONS_API EliminateUselessDiv;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Remove Multiply operation if it performs multiplication to `1`
 */

class ngraph::pass::EliminateUselessMul : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateUselessMul();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Remove Divide operation if it performs division to `1`
 */

class ngraph::pass::EliminateUselessDiv : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateUselessDiv();
};
