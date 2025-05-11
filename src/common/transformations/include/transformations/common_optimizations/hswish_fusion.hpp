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

class TRANSFORMATIONS_API HSwishFusion;
class TRANSFORMATIONS_API HSwishFusionWithReluDiv;
class TRANSFORMATIONS_API HSwishFusionWithReluMul;
class TRANSFORMATIONS_API HSwishFusionWithHSigmoid;
class TRANSFORMATIONS_API HSwishFusionWithClamp;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph (x * (min(Relu(x + 3), 6))) / 6 with a HSwish op.
 */
class ov::pass::HSwishFusionWithReluDiv : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("HSwishFusionWithReluDiv");
    HSwishFusionWithReluDiv();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph (x * (min(Relu(x + 3), 6)) * const(1/6) with a HSwish op.
 */
class ov::pass::HSwishFusionWithReluMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("HSwishFusionWithReluMul");
    HSwishFusionWithReluMul();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph x * HSigmoid(x) with a HSwish op.
 */
class ov::pass::HSwishFusionWithHSigmoid : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("HSwishFusionWithHSigmoid");
    HSwishFusionWithHSigmoid();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph (Clamp(x + 3, 0, 6) * x) with a HSwish * 6.
 */
class ov::pass::HSwishFusionWithClamp : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("HSwishFusionWithClamp");
    HSwishFusionWithClamp();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief HSwishFusion transformation replaces various sub-graphs with a HSwish op.
 */
class ov::pass::HSwishFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("HSwishFusion");
    HSwishFusion() {
        add_matcher<ov::pass::HSwishFusionWithReluDiv>();
        add_matcher<ov::pass::HSwishFusionWithReluMul>();
        add_matcher<ov::pass::HSwishFusionWithHSigmoid>();
        add_matcher<ov::pass::HSwishFusionWithClamp>();
    }
};
