// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API RandomUniformFusion;
class TRANSFORMATIONS_API RandomUniformMulFusion;
class TRANSFORMATIONS_API RandomUniformAddFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief RandomUniformMaxValFusion transformation replaces RandomUniform -> Mul subgraph
 * with a single RandomUniform node.
 *
 */
class ngraph::pass::RandomUniformMulFusion : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    RandomUniformMulFusion();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief RandomUniformMinValFusion transformation replaces RandomUniform -> Add subgraph
 * with a RandomUniform and replaces min and max const with corrected values.
 *
 */
class ngraph::pass::RandomUniformAddFusion : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    RandomUniformAddFusion();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief RandomUniformFusion transformation replaces RandomUniform with Add or Mul sub-graphs with single
 * RandomUniform.
 */
class ngraph::pass::RandomUniformFusion : public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    RandomUniformFusion() {
        add_matcher<ngraph::pass::RandomUniformMulFusion>();
        add_matcher<ngraph::pass::RandomUniformAddFusion>();
    }
};
