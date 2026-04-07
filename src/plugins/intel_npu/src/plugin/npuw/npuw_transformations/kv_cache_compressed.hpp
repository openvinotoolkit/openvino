// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::npuw {

/// Compression configuration for a single KV cache tensor (key or value).
struct KVCacheCompressionConfig {
    enum class QuantizationType { Symmetric, Asymmetric };

    QuantizationType quantization_type = QuantizationType::Asymmetric;
    ov::element::Type quantization_dt = ov::element::u8;
};

/// Independent compression parameters for key and value caches.
struct KVCacheCompressionParams {
    KVCacheCompressionConfig key;
    KVCacheCompressionConfig value;
};

// The compression passes will look for SDPA patterns,
// insert ov::op::internal::DynamicQuantize nodes on the concat outputs, redirected to result 
// and Dequantize nodes for past.key.values
// Then decompose those DynamicQuantize nodes into 
// subgraphs that expose quantization parameters as Results.  
// The added extra results like scale/zeropoints coefficients will be named 
// "DynamicQuantize/{kv_index}/present/{key|value}/scale"
// "DynamicQuantize/{kv_index}/present/{key|value}/zp"
// to be identifiable for later kv_cache_copy mechanics.  

void run_kv_cache_dynamic_quantization_passes(const std::shared_ptr<ov::Model>& model,
                                              const KVCacheCompressionParams& params);

/// Decomposition passes for ov::op::internal::DynamicQuantize.

/// V1: handcrafted symmetric-style, i8 [-127, 127]
class DecomposeDynamicQuantize : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::DecomposeDynamicQuantize");
    DecomposeDynamicQuantize();
};

/// V2: ONNX DynamicQuantizeLinear style, u8 [0, 255]
class DecomposeDynamicQuantize2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::DecomposeDynamicQuantize2");
    DecomposeDynamicQuantize2();
};

/// V3: compiler pattern style, i8 [-128, 127]
class DecomposeDynamicQuantize3 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::DecomposeDynamicQuantize3");
    DecomposeDynamicQuantize3();
};

// NOLINTNEXTLINE(readability/namespace)
}  // namespace ov::npuw
