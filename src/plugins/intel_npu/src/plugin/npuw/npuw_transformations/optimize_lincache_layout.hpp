// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {
namespace util {

// Re-lays out the depthwise-conv state of linear-attention layers (GatedDeltaNet,
// Mamba-style and LFM2 short-conv mixers) from [batch, channels, kernel] to
// [batch, kernel, channels].
//
// The state is only ever produced and consumed by the LLM pipeline itself
// (present -> past copy), so its physical layout is free to choose. With the
// channels innermost the NPU compiler can feed the Concat -> GroupConvolution
// chain in NHWC directly and the per-token transpose of the state (a slow
// element-wise permute DMA) plus its inverse on the output disappear.
//
// Graph semantics are preserved: a Transpose is inserted right after each
// cache_params.past.conv.N Parameter and right before each
// cache_params.present.conv.N Result. Tensor names stay on the model I/O.
class OptimizeLinCacheLayout : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::OptimizeLinCacheLayout");

    // Returns true when at least one conv state pair has been re-laid out.
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace util
}  // namespace npuw
}  // namespace ov
