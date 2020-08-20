// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API HSwishFusion;
class TRANSFORMATIONS_API HSwishFusionWithReluDiv;
class TRANSFORMATIONS_API HSwishFusionWithReluMul;
class TRANSFORMATIONS_API HSwishFusionWithoutRelu;


}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces various sub-graphs with a HSwish op.
 */
class ngraph::pass::HSwishFusion: public ngraph::pass::GraphRewrite {
public:
    HSwishFusion() {
        add_matcher<ngraph::pass::HSwishFusionWithReluDiv>();
        add_matcher<ngraph::pass::HSwishFusionWithReluMul>();
        add_matcher<ngraph::pass::HSwishFusionWithoutRelu>();
    }
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph (x * (min(Relu(x + 3), 6))) / 6 with a HSwish op.
 */
 class ngraph::pass::HSwishFusionWithReluDiv: public ngraph::pass::MatcherPass {
public:
    HSwishFusionWithReluDiv();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph (x * (min(Relu(x + 3), 6)) * const(1/6) with a HSwish op.
 */
 class ngraph::pass::HSwishFusionWithReluMul: public ngraph::pass::MatcherPass {
public:
    HSwishFusionWithReluMul();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph x * (min(max(x + 3, 0), 6) / 6) with a HSwish op.
 */
 class ngraph::pass::HSwishFusionWithoutRelu: public ngraph::pass::MatcherPass {
public:
    HSwishFusionWithoutRelu();
};
