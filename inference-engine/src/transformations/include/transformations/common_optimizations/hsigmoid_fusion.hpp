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

class TRANSFORMATIONS_API HSigmoidFusion;
class TRANSFORMATIONS_API HSigmoidFusionWithReluDiv;
class TRANSFORMATIONS_API HSigmoidFusionWithReluMul;
class TRANSFORMATIONS_API HSigmoidFusionWithoutRelu;
class TRANSFORMATIONS_API HSigmoidFusionWithClampMul;
class TRANSFORMATIONS_API HSigmoidFusionWithClampDiv;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph ((min(Relu(x + 3), 6)) / 6) with a HSigmoid op.
 */
class ov::pass::HSigmoidFusionWithReluDiv: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSigmoidFusionWithReluDiv();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph ((min(Relu(x + 3), 6)) * const(1/6)) with a HSigmoid op.
 */
class ov::pass::HSigmoidFusionWithReluMul: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSigmoidFusionWithReluMul();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph (min(max(x + 3, 0), 6) / 6) with a HSigmoid op.
 */
class ov::pass::HSigmoidFusionWithoutRelu: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSigmoidFusionWithoutRelu();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph (Clamp(x + 3, 0, 6) * const(1/6)) with a HSigmoid op.
 */
class ov::pass::HSigmoidFusionWithClampMul: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSigmoidFusionWithClampMul();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph (Clamp(x + 3, 0, 6) * / 6) with a HSigmoid op.
 */
class ov::pass::HSigmoidFusionWithClampDiv: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSigmoidFusionWithClampDiv();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidFusion transformation replaces various sub-graphs with a HSigmoid op.
 */
class ov::pass::HSigmoidFusion: public ov::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    HSigmoidFusion() {
        add_matcher<ov::pass::HSigmoidFusionWithReluDiv>();
        add_matcher<ov::pass::HSigmoidFusionWithReluMul>();
        add_matcher<ov::pass::HSigmoidFusionWithoutRelu>();
        add_matcher<ov::pass::HSigmoidFusionWithClampMul>();
        add_matcher<ov::pass::HSigmoidFusionWithClampDiv>();
    }
};
