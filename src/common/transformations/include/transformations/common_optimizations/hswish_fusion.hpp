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

class TRANSFORMATIONS_API HSwishFusion;
class TRANSFORMATIONS_API HSwishFusionWithReluDiv;
class TRANSFORMATIONS_API HSwishFusionWithReluMul;
class TRANSFORMATIONS_API HSwishFusionWithHSigmoid;
class TRANSFORMATIONS_API HSwishFusionWithClamp;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph (x * (min(Relu(x + 3), 6))) / 6 with a HSwish op.
 */
class ngraph::pass::HSwishFusionWithReluDiv : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("HSwishFusionWithReluDiv", "0");
    HSwishFusionWithReluDiv();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph (x * (min(Relu(x + 3), 6)) * const(1/6) with a HSwish op.
 */
class ngraph::pass::HSwishFusionWithReluMul : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("HSwishFusionWithReluMul", "0");
    HSwishFusionWithReluMul();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph x * HSigmoid(x) with a HSwish op.
 */
class ngraph::pass::HSwishFusionWithHSigmoid : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("HSwishFusionWithHSigmoid", "0");
    HSwishFusionWithHSigmoid();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph (Clamp(x + 3, 0, 6) * x) with a HSwish * 6.
 */
class ngraph::pass::HSwishFusionWithClamp : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("HSwishFusionWithClamp", "0");
    HSwishFusionWithClamp();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces various sub-graphs with a HSwish op.
 */
class ngraph::pass::HSwishFusion : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("HSwishFusion", "0");
    HSwishFusion() {
        add_matcher<ngraph::pass::HSwishFusionWithReluDiv>();
        add_matcher<ngraph::pass::HSwishFusionWithReluMul>();
        add_matcher<ngraph::pass::HSwishFusionWithHSigmoid>();
        add_matcher<ngraph::pass::HSwishFusionWithClamp>();
    }
};
