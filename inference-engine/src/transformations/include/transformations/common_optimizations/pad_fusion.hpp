// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API PadFusion;
class TRANSFORMATIONS_API PadFusionAvgPool;
class TRANSFORMATIONS_API PadFusionMaxPool;
class TRANSFORMATIONS_API PadFusionConvolution;
class TRANSFORMATIONS_API PadFusionConvolutionBackpropData;
class TRANSFORMATIONS_API PadFusionGroupConvolution;
class TRANSFORMATIONS_API PadFusionGroupConvolutionBackpropData;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief PadFusion transformation replaces following graph:
 * Pad -> AvgPool to AvgPool, under following conditions
 * - pad mode is op::PadMode::CONSTANT
 * - pad value is 0
 * - exclude_pad in AvgPool is set to false or pads_begin, pads_end are set to zero
 */
class ngraph::pass::PadFusionAvgPool: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PadFusionAvgPool();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief PadFusion transformation replaces following graph:
 * Pad -> MaxPool to MaxPool, under following conditions
 * - pad mode is op::PadMode::CONSTANT
 * - pad value is 0
 */
class ngraph::pass::PadFusionMaxPool: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PadFusionMaxPool();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief PadFusion transformation replaces following graph:
 * Pad -> Convolution to Convolution, under following conditions
 * - pad mode is op::PadMode::CONSTANT
 * - pad value is 0
 */
class ngraph::pass::PadFusionConvolution: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PadFusionConvolution();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief PadFusion transformation replaces following graph:
 * Pad -> ConvolutionBackpropData to ConvolutionBackpropData, under following conditions
 * - pad mode is op::PadMode::CONSTANT
 * - pad value is 0
 * - pads in ConvolutionBackpropData are greater than pads in Pad node
 */
class ngraph::pass::PadFusionConvolutionBackpropData: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PadFusionConvolutionBackpropData();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief PadFusion transformation replaces following graph:
 * Pad -> GroupConvolution to GroupConvolution, under following conditions
 * - pad mode is op::PadMode::CONSTANT
 * - pad value is 0
 */
class ngraph::pass::PadFusionGroupConvolution: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PadFusionGroupConvolution();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief PadFusion transformation replaces following graph:
 * Pad -> GroupConvolutionBackpropData to GroupConvolutionBackpropData, under following conditions
 * - pad mode is op::PadMode::CONSTANT
 * - pad value is 0
 * - pads in GroupConvolutionBackpropData are greater than pads in Pad node
 */
class ngraph::pass::PadFusionGroupConvolutionBackpropData: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PadFusionGroupConvolutionBackpropData();
};

class ngraph::pass::PadFusion: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    PadFusion() {
        add_matcher<ngraph::pass::PadFusionAvgPool>();
        add_matcher<ngraph::pass::PadFusionMaxPool>();
        add_matcher<ngraph::pass::PadFusionConvolution>();
        add_matcher<ngraph::pass::PadFusionConvolutionBackpropData>();
        add_matcher<ngraph::pass::PadFusionGroupConvolution>();
        add_matcher<ngraph::pass::PadFusionGroupConvolutionBackpropData>();
    }
};
