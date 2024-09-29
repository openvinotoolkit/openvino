// Copyright (C) 2018-2024 Intel Corporation
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
    OPENVINO_RTTI("ConvolutionMultiplyFusion", "0");
    ConvolutionMultiplyFusion();
};

class ov::pass::GroupConvolutionMultiplyFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GroupConvolutionMultiplyFusion", "0");
    GroupConvolutionMultiplyFusion();
};

class ov::pass::ConvolutionBackpropDataMultiplyFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvolutionBackpropDataMultiplyFusion", "0");
    ConvolutionBackpropDataMultiplyFusion();
};

class ov::pass::GroupConvolutionBackpropDataMultiplyFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GroupConvolutionBackpropDataMultiplyFusion", "0");
    GroupConvolutionBackpropDataMultiplyFusion();
};
