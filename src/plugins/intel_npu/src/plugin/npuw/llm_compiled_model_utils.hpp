// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "openvino/openvino.hpp"

namespace ov ::npuw ::util {

/*
 * special mark on nodes to be remain in high-precision for optimal processing
 */
class HighPrecisionAttr : public RuntimeAttribute {
public:
    OPENVINO_RTTI("HighPrecisionAttr", "0", RuntimeAttribute);
    ov::element::Type compute_precision_type;

    bool visit_attributes(AttributeVisitor& visitor) override {
        visitor.on_attribute("compute_precision", compute_precision_type);
        return true;
    }
};

constexpr const char* kVisualPosMasksParamName = "visual_pos_masks";
constexpr const char* kDeepstackVisualEmbedsParamName = "deepstack_visual_embeds";
constexpr const char* kTokenTypeIdsParamName = "token_type_ids";

bool has_input(const std::shared_ptr<ov::Model>& model, const std::string& name);

// Returns the KV-cache Concat that feeds an SDPA's key input through the autoregressive
// Concat->Broadcast->Reshape chain, or nullptr if the key input does not have that shape.
// Classification and reconstruction both key on this pattern, so they share one predicate
// instead of two that have to be kept in step by hand.
std::shared_ptr<ov::Node> find_kv_cache_concat(const std::shared_ptr<ov::Node>& sdpa);

// Returns true for a non-autoregressive (bidirectional encoder, e.g. BERT) text-embedding
// model: one that has ScaledDotProductAttention but none of the KV-cache concat pattern that
// the Qwen3-Embedding-style path needs in order to reconstruct a prefill/KV model.
// Used to route encoder embedders to the dedicated, KV/RoPE-free embedding path.
bool is_encoder_embedding_model(const std::shared_ptr<ov::Model>& model);

// Sanity-checks a model routed to the encoder embedding path. A bidirectional encoder is
// self-contained: it builds its own non-causal mask from `attention_mask`, works out its own
// positions and has no KV cache. Unlike the autoregressive path there is nothing to inject into
// the graph, so the only thing left to establish is that no autoregressive KV-cache input slipped
// through the classification. Throws if one did.
void validate_encoder_embedding_model(const std::shared_ptr<ov::Model>& model);

// Returns the learned absolute position-embedding table size (max_position_embeddings) of an
// encoder embedding model, found from the position_embeddings Gather's weight constant
// ([max_position_embeddings, hidden]). Returns nullopt if it can't be determined (e.g. the model
// uses a different positional scheme). The static sequence length must not exceed this, or the
// position embedding (clamped to the table) won't broadcast against the token embedding.
std::optional<uint32_t> get_max_position_embeddings(const std::shared_ptr<ov::Model>& model);

// Sliding Window Attention (SWA) layout derived from DetectAttentionMask's per-layer rt_info
// annotations (see get_layer_mask_annotations() in npuw_transformations/detect_causal_mask.hpp).
struct SwaLayout {
    uint32_t window_size = 0;            // 0 == Sliding Window Attention disabled
    std::vector<bool> layer_is_sliding;  // per-layer flag, indexed by decoder layer id
};

// Derives the KV-cache-layout implications of get_layer_mask_annotations()'s per-layer
// annotations (key absent -> Unknown/full-attention, value < 0 -> Causal, value >= 0 ->
// SlidingWindow(value)). Only enables the hybrid SWA KV-cache pipeline for a genuine hybrid
// model (at least one sliding-window layer AND at least one full-attention layer); a model
// where every layer is uniformly sliding or uniformly full attention returns window_size == 0
// (disabled), since it doesn't need per-layer KV-cache window capping.
// Throws (OPENVINO_ASSERT) if sliding-window layers disagree on window size - only a single,
// uniform window size is currently supported.
SwaLayout derive_swa_layout(const std::map<size_t, int64_t>& layer_mask_annotations);

// clang-format off
}  // namespace ov
// clang-format on
