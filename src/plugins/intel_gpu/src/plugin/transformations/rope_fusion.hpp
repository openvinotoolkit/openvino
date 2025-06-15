// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class RoPEFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("RoPEFusion");
    RoPEFusion();
};

class RoPEFusionChatGLMHF : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionChatGLMHF");
    RoPEFusionChatGLMHF();
};

}   // namespace ov::intel_gpu
