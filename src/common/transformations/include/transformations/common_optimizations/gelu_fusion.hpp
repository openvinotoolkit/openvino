// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API GeluFusion;
class TRANSFORMATIONS_API GeluFusionWithErfOne;
class TRANSFORMATIONS_API GeluFusionWithErfTwo;
class TRANSFORMATIONS_API GeluFusionWithErfThree;
class TRANSFORMATIONS_API GeluFusionWithErfFour;
class TRANSFORMATIONS_API GeluFusionWithTanh;
class TRANSFORMATIONS_API GeluFusionWithTanhNoPower;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * (0.5 * x) * (1 + erf(x / sqrt(2))) with a Gelu op.
 */
class ov::pass::GeluFusionWithErfOne : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GeluFusionWithErfOne");
    GeluFusionWithErfOne();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * 0.5 * (x * (1 + erf(x / sqrt(2)))) with a Gelu op.
 */
class ov::pass::GeluFusionWithErfTwo : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GeluFusionWithErfTwo");
    GeluFusionWithErfTwo();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * x * (0.5 * (1 + erf(x / sqrt(2)))) with a Gelu op.
 */
class ov::pass::GeluFusionWithErfThree : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GeluFusionWithErfThree");
    GeluFusionWithErfThree();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * x * (0.5 + 0.5 * erf(x * (1 / sqrt(2)))) with a Gelu op.
 */
class ov::pass::GeluFusionWithErfFour : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GeluFusionWithErfFour");
    GeluFusionWithErfFour();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * x * (0.5 * (1 + tanh([sqrt(2 / pi)] * [x + 0.044715^3]))) with a Gelu (Tanh) op.
 */
class ov::pass::GeluFusionWithTanh : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GeluFusionWithTanh");
    GeluFusionWithTanh();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * x * 0.5 * (1 + tanh((x * 0.044715 * x + 1) * x * sqrt(2 / pi))) with a Gelu (Tanh) op.
 */
class ov::pass::GeluFusionWithTanhNoPower : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GeluFusionWithTanhNoPower");
    GeluFusionWithTanhNoPower();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief GeluFusion transformation replaces various sub-graphs with a Gelu op.
 */
class ov::pass::GeluFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("GeluFusion");
    GeluFusion() {
        add_matcher<ov::pass::GeluFusionWithErfOne>();
        add_matcher<ov::pass::GeluFusionWithErfTwo>();
        add_matcher<ov::pass::GeluFusionWithErfThree>();
        add_matcher<ov::pass::GeluFusionWithErfFour>();
        add_matcher<ov::pass::GeluFusionWithTanh>();
        add_matcher<ov::pass::GeluFusionWithTanhNoPower>();
    }
};
