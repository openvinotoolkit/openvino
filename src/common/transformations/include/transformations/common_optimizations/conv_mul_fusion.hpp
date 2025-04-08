// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvolutionMultiplyFusion;
class TRANSFORMATIONS_API GroupConvolutionMultiplyFusion;
class TRANSFORMATIONS_API ConvolutionBackpropDataMultiplyFusion;
class TRANSFORMATIONS_API GroupConvolutionBackpropDataMultiplyFusion;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvolutionMultiplyFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvolutionMultiplyFusion");
    ConvolutionMultiplyFusion();
};

class ov::pass::GroupConvolutionMultiplyFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GroupConvolutionMultiplyFusion");
    GroupConvolutionMultiplyFusion();
};

class ov::pass::ConvolutionBackpropDataMultiplyFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvolutionBackpropDataMultiplyFusion");
    ConvolutionBackpropDataMultiplyFusion();
};

class ov::pass::GroupConvolutionBackpropDataMultiplyFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GroupConvolutionBackpropDataMultiplyFusion");
    GroupConvolutionBackpropDataMultiplyFusion();
};
