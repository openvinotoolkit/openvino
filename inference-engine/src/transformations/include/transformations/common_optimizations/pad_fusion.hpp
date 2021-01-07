// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API PadFusion;
class TRANSFORMATIONS_API PadFusionAvgPool;
class TRANSFORMATIONS_API PadFusionMaxPool;
class TRANSFORMATIONS_API PadFusionConvolution;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::PadFusionAvgPool: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PadFusionAvgPool();
};

class ngraph::pass::PadFusionMaxPool: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PadFusionMaxPool();
};

class ngraph::pass::PadFusionConvolution: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PadFusionConvolution();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief PadFusion transformation replaces following graph:
 * Pad -> AvgPool to AvgPool, or
 * Pad -> MaxPool to MaxPool, or
 * Pad -> Convolution to Convolution, under following conditions
 * - pad mode is op::PadMode::CONSTANT
 * - pad value is 0
 */
class ngraph::pass::PadFusion: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    PadFusion() {
        add_matcher<ngraph::pass::PadFusionAvgPool>();
        add_matcher<ngraph::pass::PadFusionMaxPool>();
        add_matcher<ngraph::pass::PadFusionConvolution>();
    }
};
