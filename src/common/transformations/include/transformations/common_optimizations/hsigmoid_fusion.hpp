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

class TRANSFORMATIONS_API HSigmoidFusion;
class TRANSFORMATIONS_API HSigmoidFusionWithReluDiv;
class TRANSFORMATIONS_API HSigmoidFusionWithReluMul;
class TRANSFORMATIONS_API HSigmoidFusionWithoutRelu;
class TRANSFORMATIONS_API HSigmoidFusionWithClampMul;
class TRANSFORMATIONS_API HSigmoidFusionWithClampDiv;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph ((min(Relu(x + 3), 6)) / 6) with a HSigmoid op.
 */
class ov::pass::HSigmoidFusionWithReluDiv : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("HSigmoidFusionWithReluDiv");
    HSigmoidFusionWithReluDiv();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph ((min(Relu(x + 3), 6)) * const(1/6)) with a HSigmoid op.
 */
class ov::pass::HSigmoidFusionWithReluMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("HSigmoidFusionWithReluMul");
    HSigmoidFusionWithReluMul();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph (min(max(x + 3, 0), 6) / 6) with a HSigmoid op.
 */
class ov::pass::HSigmoidFusionWithoutRelu : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("HSigmoidFusionWithoutRelu");
    HSigmoidFusionWithoutRelu();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph (Clamp(x + 3, 0, 6) * const(1/6)) with a HSigmoid op.
 */
class ov::pass::HSigmoidFusionWithClampMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("HSigmoidFusionWithClampMul");
    HSigmoidFusionWithClampMul();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief HSigmoidFusion transformation replaces a sub-graph (Clamp(x + 3, 0, 6) * / 6) with a HSigmoid op.
 */
class ov::pass::HSigmoidFusionWithClampDiv : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("HSigmoidFusionWithClampDiv");
    HSigmoidFusionWithClampDiv();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief HSigmoidFusion transformation replaces various sub-graphs with a HSigmoid op.
 */
class ov::pass::HSigmoidFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("HSigmoidFusion");
    HSigmoidFusion() {
        add_matcher<ov::pass::HSigmoidFusionWithReluDiv>();
        add_matcher<ov::pass::HSigmoidFusionWithReluMul>();
        add_matcher<ov::pass::HSigmoidFusionWithoutRelu>();
        add_matcher<ov::pass::HSigmoidFusionWithClampMul>();
        add_matcher<ov::pass::HSigmoidFusionWithClampDiv>();
    }
};
