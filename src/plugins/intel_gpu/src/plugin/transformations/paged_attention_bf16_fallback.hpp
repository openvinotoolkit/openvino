// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

// BF16 Q/K/V -> Convert(FP16) -> PagedAttention with FP16 cache -> Convert(original output type).
// Used as a fallback when Q/K/V are BF16 while the KV cache is explicitly FP16 (e.g., on GPUs
// without systolic array BF16 support or when mixed BF16 Q/K/V + FP16 KV cache is requested).
class PagedAttentionBF16Fallback : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::PagedAttentionBF16Fallback");
    PagedAttentionBF16Fallback();
};

}  // namespace ov::intel_gpu
