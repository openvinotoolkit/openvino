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

class TRANSFORMATIONS_API HSigmoidFusion;
class TRANSFORMATIONS_API HSigmoidFusionWithReluDiv;
class TRANSFORMATIONS_API HSigmoidFusionWithReluMul;
class TRANSFORMATIONS_API HSigmoidFusionWithoutRelu;
class TRANSFORMATIONS_API HSigmoidFusionWithClampMul;
class TRANSFORMATIONS_API HSigmoidFusionWithClampDiv;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph ((min(Relu(x + 3), 6)) / 6) with a HSigmoid op.
 */
class ngraph::pass::HSigmoidFusionWithReluDiv : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("HSigmoidFusionWithReluDiv", "0");
    HSigmoidFusionWithReluDiv();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph ((min(Relu(x + 3), 6)) * const(1/6)) with a HSigmoid op.
 */
class ngraph::pass::HSigmoidFusionWithReluMul : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("HSigmoidFusionWithReluMul", "0");
    HSigmoidFusionWithReluMul();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph (min(max(x + 3, 0), 6) / 6) with a HSigmoid op.
 */
class ngraph::pass::HSigmoidFusionWithoutRelu : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("HSigmoidFusionWithoutRelu", "0");
    HSigmoidFusionWithoutRelu();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph (Clamp(x + 3, 0, 6) * const(1/6)) with a HSigmoid op.
 */
class ngraph::pass::HSigmoidFusionWithClampMul : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("HSigmoidFusionWithClampMul", "0");
    HSigmoidFusionWithClampMul();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph (Clamp(x + 3, 0, 6) * / 6) with a HSigmoid op.
 */
class ngraph::pass::HSigmoidFusionWithClampDiv : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("HSigmoidFusionWithClampDiv", "0");
    HSigmoidFusionWithClampDiv();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces various sub-graphs with a HSigmoid op.
 */
class ngraph::pass::HSigmoidFusion : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("HSigmoidFusion", "0");
    HSigmoidFusion() {
        add_matcher<ngraph::pass::HSigmoidFusionWithReluDiv>();
        add_matcher<ngraph::pass::HSigmoidFusionWithReluMul>();
        add_matcher<ngraph::pass::HSigmoidFusionWithoutRelu>();
        add_matcher<ngraph::pass::HSigmoidFusionWithClampMul>();
        add_matcher<ngraph::pass::HSigmoidFusionWithClampDiv>();
    }
};
