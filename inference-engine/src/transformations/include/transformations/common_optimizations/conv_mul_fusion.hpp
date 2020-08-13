// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <functional>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvolutionMultiplyFusion;
class TRANSFORMATIONS_API GroupConvolutionMultiplyFusion;
class TRANSFORMATIONS_API ConvolutionBackpropDataMultiplyFusion;
class TRANSFORMATIONS_API GroupConvolutionBackpropDataMultiplyFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvolutionMultiplyFusion: public ngraph::pass::MatcherPass {
public:
    ConvolutionMultiplyFusion();
};

class ngraph::pass::GroupConvolutionMultiplyFusion: public ngraph::pass::MatcherPass {
public:
    GroupConvolutionMultiplyFusion();
};

class ngraph::pass::ConvolutionBackpropDataMultiplyFusion: public ngraph::pass::MatcherPass {
public:
    ConvolutionBackpropDataMultiplyFusion();
};

class ngraph::pass::GroupConvolutionBackpropDataMultiplyFusion: public ngraph::pass::MatcherPass {
public:
    GroupConvolutionBackpropDataMultiplyFusion();
};
