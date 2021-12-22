// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <openvino/core/ov_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class OPENVINO_API HSwishFusion;
class OPENVINO_API HSwishFusionWithReluDiv;
class OPENVINO_API HSwishFusionWithReluMul;
class OPENVINO_API HSwishFusionWithHSigmoid;
class OPENVINO_API HSwishFusionWithClamp;

}  // namespace pass
}  // namespace ngraph


/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph (x * (min(Relu(x + 3), 6))) / 6 with a HSwish op.
 */
class ngraph::pass::HSwishFusionWithReluDiv: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSwishFusionWithReluDiv();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph (x * (min(Relu(x + 3), 6)) * const(1/6) with a HSwish op.
 */
class ngraph::pass::HSwishFusionWithReluMul: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSwishFusionWithReluMul();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph x * HSigmoid(x) with a HSwish op.
 */
class ngraph::pass::HSwishFusionWithHSigmoid: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSwishFusionWithHSigmoid();
};


/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph (Clamp(x + 3, 0, 6) * x) with a HSwish * 6.
 */
class ngraph::pass::HSwishFusionWithClamp: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSwishFusionWithClamp();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces various sub-graphs with a HSwish op.
 */
class ngraph::pass::HSwishFusion: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    HSwishFusion() {
        add_matcher<ngraph::pass::HSwishFusionWithReluDiv>();
        add_matcher<ngraph::pass::HSwishFusionWithReluMul>();
        add_matcher<ngraph::pass::HSwishFusionWithHSigmoid>();
        add_matcher<ngraph::pass::HSwishFusionWithClamp>();
    }
};
