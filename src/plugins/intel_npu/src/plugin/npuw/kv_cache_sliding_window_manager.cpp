// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_sliding_window_manager.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "logging.hpp"
#include "npuw_transformations/detect_causal_mask.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "util.hpp"

ov::npuw::util::SwaLayout ov::npuw::util::detect_swa_layout(const std::shared_ptr<ov::Model>& model) {
    std::vector<int64_t> layer_mask_annotations;
    std::vector<bool> layer_has_annotation;
    size_t num_annotated_layers = 0;
    const size_t max_reasonable_layer_idx = model->get_ordered_ops().size();

    for (const auto& node : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node);
        if (!sdpa) {
            continue;
        }
        size_t layer_idx = 0;
        if (!try_parse_self_attn_layer_idx(sdpa->get_friendly_name(), layer_idx)) {
            continue;
        }
        OPENVINO_ASSERT(layer_idx < max_reasonable_layer_idx,
                        "NPUW SWA: unreasonable layer index ",
                        layer_idx,
                        " in SDPA name '",
                        sdpa->get_friendly_name(),
                        "' (number of graph ops is ",
                        max_reasonable_layer_idx,
                        ").");

        // Every parseable layer must be represented, even when annotation is absent.
        // Missing annotation is interpreted as non-SWA (full/causal) later.
        if (layer_idx >= layer_mask_annotations.size()) {
            layer_mask_annotations.resize(layer_idx + 1, 0);
            layer_has_annotation.resize(layer_idx + 1, false);
        }

        const auto& rt_info = sdpa->get_rt_info();
        const auto it = rt_info.find(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
        if (it == rt_info.end()) {
            continue;
        }

        const int64_t encoded = it->second.as<int64_t>();
        if (layer_has_annotation[layer_idx]) {
            OPENVINO_ASSERT(layer_mask_annotations[layer_idx] == encoded,
                            "NPUW SWA: conflicting SDPA mask annotations for layer ",
                            layer_idx,
                            " (",
                            layer_mask_annotations[layer_idx],
                            " vs ",
                            encoded,
                            ").");
        } else {
            layer_mask_annotations[layer_idx] = encoded;
            layer_has_annotation[layer_idx] = true;
            ++num_annotated_layers;
        }
    }

    SwaLayout layout;

    if (num_annotated_layers == 0) {
        LOG_DEBUG("[SWA] No per-layer mask annotations found; Sliding Window Attention is disabled.");
        return layout;
    }

    // Layers absent from the annotation set are treated as full-attention.
    const size_t num_layers = layer_mask_annotations.size();

    std::vector<bool> layer_is_sliding(num_layers, false);
    int64_t detected_window = 0;
    bool has_detected_window = false;
    bool has_sliding = false;
    bool has_full = false;

    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        if (layer_has_annotation[layer_idx] && layer_mask_annotations[layer_idx] >= 0) {
            const int64_t encoded = layer_mask_annotations[layer_idx];
            layer_is_sliding[layer_idx] = true;
            has_sliding = true;
            if (!has_detected_window) {
                detected_window = encoded;
                has_detected_window = true;
            } else {
                OPENVINO_ASSERT(detected_window == encoded,
                                "NPUW SWA: inconsistent sliding-window sizes detected across layers (",
                                detected_window,
                                " vs ",
                                encoded,
                                "). Only a single, uniform window size is currently supported.");
            }
        } else {
            has_full = true;
        }
    }

    // Only enable the hybrid SWA pipeline for a genuine hybrid model: at least one SWA
    // layer AND at least one non-SWA (full/causal) layer. A model where every layer is uniformly
    // SWA (or uniformly non-SWA) doesn't need per-layer KV-cache window capping.
    if (!has_sliding || !has_full) {
        LOG_DEBUG("[SWA] Not a genuine hybrid model (has_swa=" << has_sliding << ", has_non_swa=" << has_full
                                                               << "); SWA support disabled.");
        return layout;
    }

    OPENVINO_ASSERT(has_detected_window && detected_window > 0,
                    "NPUW SWA: invalid SWA window size detected: ",
                    detected_window);
    OPENVINO_ASSERT(detected_window <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                    "NPUW SWA: SWA window size exceeds uint32_t range: ",
                    detected_window);

    layout.window_size = static_cast<uint32_t>(detected_window);
    layout.layer_is_sliding = std::move(layer_is_sliding);

    std::string pattern;
    pattern.reserve(layout.layer_is_sliding.size());
    size_t num_sliding = 0;
    for (const bool is_sliding : layout.layer_is_sliding) {
        pattern.push_back(is_sliding ? 'S' : 'F');
        num_sliding += is_sliding ? 1 : 0;
    }
    LOG_INFO("[SWA] Sliding Window Attention is ENABLED: window_size="
             << layout.window_size << ", " << layout.layer_is_sliding.size() << " layers total, " << num_sliding
             << " sliding-window layer(s).");
    LOG_DEBUG("[SWA] Layer pattern (S=sliding, F=full attention): " << pattern);

    return layout;
}
