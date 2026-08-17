// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Analysis pass: detects the attention mask type of each SDPA node in the model by
// inspecting its mask-construction subgraph (Range/LessEqual/Greater/BitwiseAnd) or
// its is_causal attribute, and annotates the node's rt_info[NPUW_SDPA_MASK_RT_KEY]
// accordingly (see the encoding documented there). A node may only be annotated
// once -- a matcher that would overwrite an already-annotated node with a
// conflicting kind asserts instead, since that indicates genuinely contradictory
// evidence (e.g. an is_causal=true SDPA fed an explicit sliding-window mask), not
// just two matchers agreeing.
//
// Must run before SDPA decomposition (OptimizeValueTensors) so the SDPA nodes it
// annotates still exist.
class DetectAttentionMask : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::DetectAttentionMask");
    DetectAttentionMask() = default;

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

void log_detected_masks(const std::shared_ptr<ov::Model>& model);

// rt_info key written by DetectAttentionMask onto each ScaledDotProductAttention
// node, propagated onto the decomposed Add(QK, mask) node by
// ScaledDotProductAttentionDecomposition::decompose() (via copy_runtime_info), and
// read directly by HostFlashAttention::from().
//
// Value type: int64_t, encoding both the mask kind and (for sliding window) its
// window size in a single slot:
//   * key absent   -> Unknown (no recognized mask pattern, e.g. full/bidirectional
//                     attention)
//   * value <  0   -> Causal (equivalent to a sliding window whose size covers the
//                     whole context, i.e. "infinite window")
//   * value >= 0   -> SlidingWindow, value is the window size
static constexpr const char* NPUW_SDPA_MASK_RT_KEY = "npuw_sdpa_mask_type";

// Sentinel value written for the Causal case -- see NPUW_SDPA_MASK_RT_KEY above.
static constexpr int64_t NPUW_SDPA_MASK_CAUSAL = -1;

}  // namespace ov::npuw
