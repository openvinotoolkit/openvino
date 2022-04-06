// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/pass.hpp>
#include <ngraph/util.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvStridesPropagation;
class TRANSFORMATIONS_API SupportedNodesStridesPropagation;
class TRANSFORMATIONS_API UnsupportedNodesStridesPropagation;
class TRANSFORMATIONS_API StridesOptimization;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvStridesPropagation either propagates stride (greater than 1) from Convolution up through the graph
 * or inserts pooling between current node and its consumers if the consumers have different StridesProp attributes.
 * Strides can be propagated if Convolution kernel is {1, 1, ...}
 */
class ngraph::pass::ConvStridesPropagation : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvStridesPropagation", "0");
    ConvStridesPropagation();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SupportedNodesStridesPropagation either propagates stride (greater than 1) from current node up through the
 * graph or inserts pooling between current node and its consumers if the consumers have different StridesProp
 * attributes.
 */
class ngraph::pass::SupportedNodesStridesPropagation : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SupportedNodesStridesPropagation", "0");
    SupportedNodesStridesPropagation();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief UnsupportedNodesStridesPropagation inserts pooling between current node and its consumers
 * if the consumers have different StridesProp attributes.
 */
class ngraph::pass::UnsupportedNodesStridesPropagation : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("UnsupportedNodesStridesPropagation", "0");
    UnsupportedNodesStridesPropagation();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief StridesOptimization transformation works backward on function and propagates strides up through the graph if
 * possible
 */
class ngraph::pass::StridesOptimization : public ngraph::pass::BackwardGraphRewrite {
public:
    OPENVINO_RTTI("StridesOptimization", "0");
    StridesOptimization() {
        add_matcher<ngraph::pass::ConvStridesPropagation>();
        add_matcher<ngraph::pass::SupportedNodesStridesPropagation>();
        add_matcher<ngraph::pass::UnsupportedNodesStridesPropagation>();
    }
};
