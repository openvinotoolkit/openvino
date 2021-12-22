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

class OPENVINO_API HSigmoidFusion;
class OPENVINO_API HSigmoidFusionWithReluDiv;
class OPENVINO_API HSigmoidFusionWithReluMul;
class OPENVINO_API HSigmoidFusionWithoutRelu;
class OPENVINO_API HSigmoidFusionWithClampMul;
class OPENVINO_API HSigmoidFusionWithClampDiv;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph ((min(Relu(x + 3), 6)) / 6) with a HSigmoid op.
 */
class ngraph::pass::HSigmoidFusionWithReluDiv: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSigmoidFusionWithReluDiv();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph ((min(Relu(x + 3), 6)) * const(1/6)) with a HSigmoid op.
 */
class ngraph::pass::HSigmoidFusionWithReluMul: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSigmoidFusionWithReluMul();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph (min(max(x + 3, 0), 6) / 6) with a HSigmoid op.
 */
class ngraph::pass::HSigmoidFusionWithoutRelu: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSigmoidFusionWithoutRelu();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph (Clamp(x + 3, 0, 6) * const(1/6)) with a HSigmoid op.
 */
class ngraph::pass::HSigmoidFusionWithClampMul: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSigmoidFusionWithClampMul();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph (Clamp(x + 3, 0, 6) * / 6) with a HSigmoid op.
 */
class ngraph::pass::HSigmoidFusionWithClampDiv: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSigmoidFusionWithClampDiv();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces various sub-graphs with a HSigmoid op.
 */
class ngraph::pass::HSigmoidFusion: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    HSigmoidFusion() {
        add_matcher<ngraph::pass::HSigmoidFusionWithReluDiv>();
        add_matcher<ngraph::pass::HSigmoidFusionWithReluMul>();
        add_matcher<ngraph::pass::HSigmoidFusionWithoutRelu>();
        add_matcher<ngraph::pass::HSigmoidFusionWithClampMul>();
        add_matcher<ngraph::pass::HSigmoidFusionWithClampDiv>();
    }
};
