// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API GeluFusion;
class TRANSFORMATIONS_API GeluFusionWithErfOne;
class TRANSFORMATIONS_API GeluFusionWithErfTwo;
class TRANSFORMATIONS_API GeluFusionWithErfThree;
class TRANSFORMATIONS_API GeluFusionWithTanh;

}  // namespace pass
}  // namespace ngraph

namespace ov {
namespace pass {

class TRANSFORMATIONS_API GeluFusionWithErfFour;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * (0.5 * x) * (1 + erf(x / sqrt(2))) with a Gelu op.
 */
class ngraph::pass::GeluFusionWithErfOne : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("GeluFusionWithErfOne", "0");
    GeluFusionWithErfOne();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * 0.5 * (x * (1 + erf(x / sqrt(2)))) with a Gelu op.
 */
class ngraph::pass::GeluFusionWithErfTwo : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("GeluFusionWithErfTwo", "0");
    GeluFusionWithErfTwo();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * x * (0.5 * (1 + erf(x / sqrt(2)))) with a Gelu op.
 */
class ngraph::pass::GeluFusionWithErfThree : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("GeluFusionWithErfThree", "0");
    GeluFusionWithErfThree();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * x * (0.5 + 0.5 * erf(x * (1 / sqrt(2)))) with a Gelu op.
 */
class ov::pass::GeluFusionWithErfFour : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GeluFusionWithErfFour", "0");
    GeluFusionWithErfFour();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * x * (0.5 * (1 + tanh([sqrt(2 / pi)] * [x + 0.044715^3]))) with a Gelu (Tanh) op.
 */
class ngraph::pass::GeluFusionWithTanh : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("GeluFusionWithTanh", "0");
    GeluFusionWithTanh();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces various sub-graphs with a Gelu op.
 */
class ngraph::pass::GeluFusion : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("GeluFusion", "0");
    GeluFusion() {
        add_matcher<ngraph::pass::GeluFusionWithErfOne>();
        add_matcher<ngraph::pass::GeluFusionWithErfTwo>();
        add_matcher<ngraph::pass::GeluFusionWithErfThree>();
        add_matcher<ov::pass::GeluFusionWithErfFour>();
        add_matcher<ngraph::pass::GeluFusionWithTanh>();
    }
};
