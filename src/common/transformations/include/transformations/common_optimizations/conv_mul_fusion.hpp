// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <functional>

#include <openvino/core/visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class OPENVINO_API ConvolutionMultiplyFusion;
class OPENVINO_API GroupConvolutionMultiplyFusion;
class OPENVINO_API ConvolutionBackpropDataMultiplyFusion;
class OPENVINO_API GroupConvolutionBackpropDataMultiplyFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvolutionMultiplyFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvolutionMultiplyFusion();
};

class ngraph::pass::GroupConvolutionMultiplyFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GroupConvolutionMultiplyFusion();
};

class ngraph::pass::ConvolutionBackpropDataMultiplyFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvolutionBackpropDataMultiplyFusion();
};

class ngraph::pass::GroupConvolutionBackpropDataMultiplyFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GroupConvolutionBackpropDataMultiplyFusion();
};
