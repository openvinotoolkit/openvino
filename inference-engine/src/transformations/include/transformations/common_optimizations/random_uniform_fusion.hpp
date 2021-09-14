// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API RandomUniformFusion;
class TRANSFORMATIONS_API RandomUniformMulAddFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief RandomUniformMulAddFusion transformation replaces RandomUniform -> Add or
 * RandomUniform -> Mul subgraph with a RandomUniform and replaces min and max const
 * with corrected values.
 */
class ngraph::pass::RandomUniformMulAddFusion : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    RandomUniformMulAddFusion();
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
        add_matcher<ngraph::pass::RandomUniformMulAddFusion>();
    }
};
