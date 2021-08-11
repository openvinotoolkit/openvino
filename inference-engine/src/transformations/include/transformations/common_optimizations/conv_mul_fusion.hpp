// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <functional>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvolutionMultiplyFusion;
class TRANSFORMATIONS_API GroupConvolutionMultiplyFusion;
class TRANSFORMATIONS_API ConvolutionBackpropDataMultiplyFusion;
class TRANSFORMATIONS_API GroupConvolutionBackpropDataMultiplyFusion;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvolutionMultiplyFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvolutionMultiplyFusion();
};

class ov::pass::GroupConvolutionMultiplyFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GroupConvolutionMultiplyFusion();
};

class ov::pass::ConvolutionBackpropDataMultiplyFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvolutionBackpropDataMultiplyFusion();
};

class ov::pass::GroupConvolutionBackpropDataMultiplyFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GroupConvolutionBackpropDataMultiplyFusion();
};
