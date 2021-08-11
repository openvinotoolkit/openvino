// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API GeluFusion;
class TRANSFORMATIONS_API GeluFusionWithErfOne;
class TRANSFORMATIONS_API GeluFusionWithErfTwo;
class TRANSFORMATIONS_API GeluFusionWithErfThree;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * (0.5 * x) * (1 + erf(x / sqrt(2))) with a Gelu op.
 */
class ov::pass::GeluFusionWithErfOne : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GeluFusionWithErfOne();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * 0.5 * (x * (1 + erf(x / sqrt(2))) with a Gelu op.
 */
class ov::pass::GeluFusionWithErfTwo : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GeluFusionWithErfTwo();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * x * (0.5 * (1 + erf(x / sqrt(2))) with a Gelu op.
 */
class ov::pass::GeluFusionWithErfThree : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GeluFusionWithErfThree();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces various sub-graphs with a Gelu op.
 */
class ov::pass::GeluFusion : public ov::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    GeluFusion() {
        add_matcher<ov::pass::GeluFusionWithErfOne>();
        add_matcher<ov::pass::GeluFusionWithErfTwo>();
        add_matcher<ov::pass::GeluFusionWithErfThree>();
    }
};
