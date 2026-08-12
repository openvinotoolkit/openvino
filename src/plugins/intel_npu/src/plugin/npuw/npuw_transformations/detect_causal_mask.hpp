// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

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

// rt_info key written by AnnotatePerSDPAMaskType and read by HostFlashAttention::from().
// Value type: int, corresponding to MaskInfo::MaskType.
static constexpr const char* NPUW_SDPA_MASK_TYPE_RT_KEY = "npuw_sdpa_mask_type";

// Pre-partitioning pass: annotates each decomposed-SDPA's Add(QK, mask) node in the
// model with its individual mask type via rt_info[NPUW_SDPA_MASK_TYPE_RT_KEY].
//
// This enables per-layer mask-skipping decisions inside HostFlashAttention::from()
// for mixed SWA + global-attention models (e.g. Gemma-4 E2B/E4B): global-attention
// ATTN subgraphs can keep mask skipping enabled even when SWA layers are present.
//
// Must be run on the whole model BEFORE partitioning so the annotation is carried
// into the isolated ATTN subgraphs (the Add node object is shared, not cloned).
// Never modifies the graph structure; run_on_model always returns false.
class AnnotatePerSDPAMaskType : public ov::pass::ModelPass {
public:
    struct Annotation {
        std::string add_node_name;
        MaskInfo::MaskType mask_type = MaskInfo::MaskType::Unknown;
    };

    OPENVINO_MODEL_PASS_RTTI("ov::npuw::AnnotatePerSDPAMaskType");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

    // Collected per-SDPA mask types from the most recent run_on_model() call.
    const std::vector<Annotation>& get_annotations() const {
        return m_annotations;
    }

    // Convenience helper: returns only mask types in traversal order.
    std::vector<MaskInfo::MaskType> get_mask_types() const {
        std::vector<MaskInfo::MaskType> mask_types;
        mask_types.reserve(m_annotations.size());
        for (const auto& annotation : m_annotations)
            mask_types.push_back(annotation.mask_type);
        return mask_types;
    }

private:
    std::vector<Annotation> m_annotations;
};

}  // namespace ov::npuw
