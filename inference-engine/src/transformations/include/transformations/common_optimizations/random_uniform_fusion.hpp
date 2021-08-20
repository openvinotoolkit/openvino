// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API RandomUniformFusion;
class TRANSFORMATIONS_API RandomUniformMaxValFusion;
class TRANSFORMATIONS_API RandomUniformMinValFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief RandomUniformMaxValFusion transformation replaces RandomUniform -> Mul subgraph
 * with a single RandomUniform node.
 *
 * * Restrictions:
 * RandomUniform node should have floating output type and input max value = 1.0.
 */
class ngraph::pass::RandomUniformMaxValFusion : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    RandomUniformMaxValFusion();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief RandomUniformMinValFusion transformation replaces RandomUniform -> Add subgraph
 * with a RandomUniform and replaces min and max const with corrected values.
 *
 * * Restrictions:
 * RandomUniform node should have floating output type and input min value = 0.0.
 */
class ngraph::pass::RandomUniformMinValFusion : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    RandomUniformMinValFusion();
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
        add_matcher<ngraph::pass::RandomUniformMaxValFusion>();
        add_matcher<ngraph::pass::RandomUniformMinValFusion>();
    }
};
