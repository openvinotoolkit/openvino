// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

class ClampFP16Output: public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ClampFP16Output");
    ClampFP16Output();
};

/**
 * @brief This transformation adds Clamp primitive between MatMul and Softmax operation
 * which is targeting some transformer based models (mainly Stable Diffusion) which may have an fp16 overflow
 * on MatMul output tensor which could lead to Inf/Nan values on the model output.
 * We assume that Clamp operation handling costs almost nothing from the performance perspective as it's supposed to be fused to MatMul later
 */
class ClampFP16OutputSoftmaxMatcher: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ClampFP16OutputSoftmaxMatcher");
    ClampFP16OutputSoftmaxMatcher();
};

class ClampFP16OutputRMSMatcher: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ClampFP16OutputRMSMatcher");
    ClampFP16OutputRMSMatcher();
};

}   // namespace ov::intel_gpu
