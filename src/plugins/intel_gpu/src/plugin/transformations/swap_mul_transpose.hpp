// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

class SwapMulTranspose: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::SwapMulTranspose");
    SwapMulTranspose();
};

class VariadicSplitMulFusion: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::VariadicSplitMulFusion");
    VariadicSplitMulFusion();
};

}   // namespace ov::intel_gpu
