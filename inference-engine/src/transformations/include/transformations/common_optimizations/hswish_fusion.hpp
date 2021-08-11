// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API HSwishFusion;
class TRANSFORMATIONS_API HSwishFusionWithReluDiv;
class TRANSFORMATIONS_API HSwishFusionWithReluMul;
class TRANSFORMATIONS_API HSwishFusionWithHSigmoid;

}  // namespace pass
}  // namespace ov


/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph (x * (min(Relu(x + 3), 6))) / 6 with a HSwish op.
 */
class ov::pass::HSwishFusionWithReluDiv: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSwishFusionWithReluDiv();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph (x * (min(Relu(x + 3), 6)) * const(1/6) with a HSwish op.
 */
class ov::pass::HSwishFusionWithReluMul: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSwishFusionWithReluMul();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph x * HSigmoid(x) with a HSwish op.
 */
class ov::pass::HSwishFusionWithHSigmoid: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSwishFusionWithHSigmoid();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces various sub-graphs with a HSwish op.
 */
class ov::pass::HSwishFusion: public ov::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    HSwishFusion() {
        add_matcher<ov::pass::HSwishFusionWithReluDiv>();
        add_matcher<ov::pass::HSwishFusionWithReluMul>();
        add_matcher<ov::pass::HSwishFusionWithHSigmoid>();
    }
};
