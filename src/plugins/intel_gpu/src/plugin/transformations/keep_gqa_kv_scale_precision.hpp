// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

// Marks GroupQueryAttention KV dequant scales (k_scale/v_scale) precision-sensitive so
// ConvertPrecision keeps them fp32. GPU runs ConvertPrecision before the GQA op is decomposed,
// and the intact op requires fp32 scales (com.microsoft spec). Applies to a quantized KV cache only.
class KeepGQAKVScalePrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::KeepGQAKVScalePrecision");
    KeepGQAKVScalePrecision();
};

}  // namespace ov::intel_gpu
