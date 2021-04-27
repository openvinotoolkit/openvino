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
class TRANSFORMATIONS_API StridePropagation;
class TRANSFORMATIONS_API StrideOptimization;

}  // namespace pass
}  // namespace ngraph


class ngraph::pass::ConvStridePropagation: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvStridePropagation();
};

class ngraph::pass::StridePropagation: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    StridePropagation();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief StrideOptimization transformation is a FunctionPass
 * that propagates stride (greater than 1) from Convolution
 * up through the graph (namely Relu, Maximum, Mul, Add and Conv operators)
 */

class ngraph::pass::StrideOptimization: public ngraph::pass::BackwardGraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    StrideOptimization() {
        add_matcher<ngraph::pass::ConvStridePropagation>();
        add_matcher<ngraph::pass::StridePropagation>();
    }
};
