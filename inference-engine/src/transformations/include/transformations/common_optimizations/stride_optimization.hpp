// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <transformations_visibility.hpp>
#include <ngraph/util.hpp>
#include <ngraph/pass/pass.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvStridePropagation;
class TRANSFORMATIONS_API SupportedNodesStridePropagation;
class TRANSFORMATIONS_API OtherNodesStridePropagation;
class TRANSFORMATIONS_API StrideOptimization;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvStridePropagation either propagates stride (greater than 1) from Convolution up through the graph
 * or inserts pooling between current node and its consumers if the consumers have different StridesProp attributes.
 * Stride can be propagated if Convolution kernel is {1, 1, ...}
 */
class ngraph::pass::ConvStridePropagation: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvStridePropagation();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SupportedNodesStridePropagation either propagates stride (greater than 1) from current node up through the graph
 * or inserts pooling between current node and its consumers if the consumers have different StridesProp attributes.
 */
class ngraph::pass::SupportedNodesStridePropagation: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SupportedNodesStridePropagation();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief OtherNodesStridePropagation inserts pooling between current node and its consumers
 * if the consumers have different StridesProp attributes.
 */
class ngraph::pass::OtherNodesStridePropagation: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    OtherNodesStridePropagation();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief StrideOptimization transformation works backward on function and propagates strides up through the graph if possible
 */
class ngraph::pass::StrideOptimization: public ngraph::pass::BackwardGraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    StrideOptimization() {
        add_matcher<ngraph::pass::ConvStridePropagation>();
        add_matcher<ngraph::pass::SupportedNodesStridePropagation>();
        add_matcher<ngraph::pass::OtherNodesStridePropagation>();
    }
};
