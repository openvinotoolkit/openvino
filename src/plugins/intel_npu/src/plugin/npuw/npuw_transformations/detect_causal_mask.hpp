// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Detected attention mask type and (for sliding window) its window size.
struct MaskInfo {
    // No recognized mask pattern (e.g. full attention), plain causal, or
    // causal + sliding-window (local attention).
    enum class MaskType : int { Unknown = 0, Causal, SlidingWindow };

    // Value-initialized to MaskType::Unknown (== 0).
    MaskType mask_type{};
    // Valid only when mask_type == SlidingWindow.
    int64_t window_size = 0;
};

// Analysis pass: detects the attention mask type of a model by inspecting the
// mask-construction subgraph (Range/LessEqual/Greater/BitwiseAnd) and the SDPA
// is_causal attribute. It never modifies the model (run_on_model returns false),
// so the result is retrieved via get_mask_info() after running.
class DetectAttentionMask : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::DetectAttentionMask");
    DetectAttentionMask() = default;

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

    const MaskInfo& get_mask_info() const {
        return m_mask_info;
    }

private:
    MaskInfo m_mask_info;
};

}  // namespace ov::npuw
