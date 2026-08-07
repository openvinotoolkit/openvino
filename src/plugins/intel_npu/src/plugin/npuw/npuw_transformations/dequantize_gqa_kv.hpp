// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Makes a quantized-KV (int8) com.microsoft GroupQueryAttention lowerable on the NPU by dequantizing its
// KV cache *around* the op while keeping the op itself intact and float.
//
// The NPU (vpux) compiler has a native fused lowering for the FLOAT GroupQueryAttention op — the same one the
// fp16 KV-cache path uses — but not for the quantized variant: carrying the int8 KV + per-channel scales into
// the intact op makes vpux reconstruct the packed [B, num_heads+2*kv_num_heads, S, H] QKV and try to apply the
// per-KV-head scale onto it, which fails ("non broadcastable dimensions 60 and 10"). This pass instead:
//
//   past_key(i8) --Convert--Multiply(scale[1,kv,1,H])--> float --+
//                                                                 |--> [ intact FLOAT GroupQueryAttention ] --+
//   past_value(i8) --Convert--Multiply(scale)---------> float ---+                                           |
//   present_key(float) --Multiply(1/scale)-Round-Clamp-Convert--> i8   (requant, restores the i8 KV I/O) <---+
//
// The dequant Multiply is applied ONLY to the op's separate KV inputs (3/4), each [B, kv_num_heads, S, H], so
// the scale broadcasts cleanly and never touches the packed QKV. The rebuilt op carries no quant metadata
// (kv_cache_bit_width=0, k/v_quant_type="NONE") so it routes to the known-good float lowering. Each (layer, K/V)
// scale is materialized as a fresh single-reader Constant (data-copy) so NPUW's FOLD pass can build a complete
// per-instance scale bank. Float GQA is left untouched.
class DequantizeGQAKVCache : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::DequantizeGQAKVCache");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
