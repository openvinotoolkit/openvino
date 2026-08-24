// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_sliding_window_manager.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "llm_compiled_model_utils.hpp"
#include "logging.hpp"
#include "npuw_transformations/detect_causal_mask.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/itensor.hpp"
#include "util.hpp"

namespace {

struct MaskView {
    uint32_t row_dim = 0;
    uint32_t col_dim = 0;
    uint32_t past_width = 0;
    uint32_t row_pad = 0;
    float* data = nullptr;
};

MaskView get_mask_view(const ov::SoPtr<ov::ITensor>& mask_tensor,
                       uint32_t num_real_new_tokens,
                       const char* caller_name) {
    OPENVINO_ASSERT(mask_tensor->get_element_type() == ov::element::f32,
                    caller_name,
                    ": Attention mask tensor is expected to be f32, got: ",
                    mask_tensor->get_element_type());
    const auto& shape = mask_tensor->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, caller_name, ": Attention mask tensor rank must be >= 2, got shape: ", shape);

    const uint32_t row_dim = static_cast<uint32_t>(shape[shape.size() - 2]);
    const uint32_t col_dim = static_cast<uint32_t>(shape[shape.size() - 1]);
    OPENVINO_ASSERT(col_dim >= row_dim,
                    caller_name,
                    ": attention mask key axis (",
                    col_dim,
                    ") must be >= query axis (",
                    row_dim,
                    ")");
    OPENVINO_ASSERT(num_real_new_tokens <= row_dim,
                    caller_name,
                    ": num_real_new_tokens (",
                    num_real_new_tokens,
                    ") exceeds query axis (",
                    row_dim,
                    ")");

    MaskView mv;
    mv.row_dim = row_dim;
    mv.col_dim = col_dim;
    mv.past_width = col_dim - row_dim;
    mv.row_pad = row_dim - num_real_new_tokens;
    mv.data = mask_tensor->data<float>();
    return mv;
}

}  // namespace

ov::npuw::util::SwaLayout ov::npuw::util::detect_swa_layout(const std::shared_ptr<ov::Model>& model) {
    // Read back per-layer SDPA mask annotations (see NPUW_SDPA_MASK_RT_KEY).
    // Missing entries are treated as full/causal layers.
    std::vector<int64_t> layer_mask_annotations;
    std::vector<bool> layer_has_annotation;
    size_t num_annotated_layers = 0;

    for (const auto& node : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node);
        if (!sdpa) {
            continue;
        }
        size_t layer_idx = 0;
        if (!try_parse_self_attn_layer_idx(sdpa->get_friendly_name(), layer_idx)) {
            continue;
        }
        const auto& rt_info = sdpa->get_rt_info();
        const auto it = rt_info.find(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
        if (it == rt_info.end()) {
            continue;
        }

        const int64_t encoded = it->second.as<int64_t>();
        if (layer_idx >= layer_mask_annotations.size()) {
            layer_mask_annotations.resize(layer_idx + 1, 0);
            layer_has_annotation.resize(layer_idx + 1, false);
        }

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

    // Only enable the hybrid SWA pipeline for a genuine hybrid model: at least one sliding-window
    // layer AND at least one full/causal-attention layer. A model where every layer is uniformly
    // sliding (or uniformly full attention) doesn't need per-layer KV-cache window capping.
    if (!has_sliding || !has_full) {
        LOG_DEBUG("[SWA] Not a genuine hybrid model (has_sliding=" << has_sliding << ", has_full=" << has_full
                                                                   << "); Sliding Window Attention support disabled.");
        return layout;
    }

    OPENVINO_ASSERT(has_detected_window && detected_window > 0,
                    "NPUW SWA: invalid sliding window size detected: ",
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

void ov::npuw::util::fill_causal_sliding_mask(ov::SoPtr<ov::ITensor> mask_tensor,
                                              uint32_t num_stored_tokens_before,
                                              uint32_t num_real_new_tokens,
                                              uint32_t window_size) {
    const auto mask_view = get_mask_view(mask_tensor, num_real_new_tokens, "fill_causal_sliding_mask");
    OPENVINO_ASSERT(window_size > 0, "fill_causal_sliding_mask: window_size must be > 0");
    OPENVINO_ASSERT(mask_view.past_width > 0,
                    "fill_causal_sliding_mask: past_width is zero; sliding-window mask expects non-zero past width");

    const uint32_t P = num_stored_tokens_before;

    constexpr float kAttend = 0.0f;
    const float kMasked = static_cast<float>(std::numeric_limits<ov::float16>::lowest());

    for (uint32_t row = 0; row < mask_view.row_dim; ++row) {
        float* row_ptr = mask_view.data + static_cast<size_t>(row) * mask_view.col_dim;

        // Past columns: c in [0, past_width). The past K/V buffer is maintained by
        // write_swa_kv_slice_circular(): physical slot c always holds whichever absolute
        // token position last landed there via `p % past_width` - no data is ever shifted.
        // While the window has
        // not yet saturated (P < past_width), writes fill physical slots strictly in arrival
        // order 0, 1, 2, ... - so the valid prefix is LEFT-aligned at [0, P) (column c holds
        // absolute position c), and columns >= P are still-uninitialized garbage. Once
        // P >= past_width (saturated at least once), every physical slot has been written at
        // least once (all valid), but which absolute position slot c currently holds depends on
        // how far the wrap-around write cursor `r = P % past_width` has progressed: slots >= r
        // hold the most recently *completed* lap (abs = P - r + c - past_width), slots < r hold
        // the lap currently in progress (abs = P - r + c).
        const int64_t row_local = static_cast<int64_t>(row) - static_cast<int64_t>(mask_view.row_pad);
        const int64_t q = static_cast<int64_t>(P) + row_local;  // this row's own absolute position
        const uint32_t r = P % mask_view.past_width;
        for (uint32_t c = 0; c < mask_view.past_width; ++c) {
            bool valid;
            int64_t abs_pos;
            if (P >= mask_view.past_width) {
                valid = true;
                abs_pos = (c < r) ? (static_cast<int64_t>(P) - r + c)
                                  : (static_cast<int64_t>(P) - r + c - mask_view.past_width);
            } else {
                valid = c < P;
                abs_pos = c;
            }
            const bool causal = abs_pos <= q;
            const bool window_ok = (q - abs_pos) < static_cast<int64_t>(window_size);
            const bool attend = valid && causal && window_ok;
            row_ptr[c] = attend ? kAttend : kMasked;
        }

        // Current-chunk diagonal columns: local_c in [0, row_dim), mapped to c = past_width +
        // local_c. Both axes share the same row_pad right-alignment offset, so it cancels
        // identically in both the causal and window comparisons below - raw indices suffice.
        for (uint32_t local_c = 0; local_c < mask_view.row_dim; ++local_c) {
            const bool valid_key = local_c >= mask_view.row_pad;
            const bool causal = local_c <= row;
            const bool window_ok = causal && (row - local_c) < window_size;
            const bool attend = valid_key && causal && window_ok;
            row_ptr[mask_view.past_width + local_c] = attend ? kAttend : kMasked;
        }
    }
}

void ov::npuw::util::fill_attention_masks(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                          const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                                          uint32_t num_stored_tokens_before,
                                          uint32_t num_real_new_tokens,
                                          uint32_t window_size,
                                          const int64_t* token_type_ids_real) {
    const auto it = in_ports.find(ov::npuw::util::kSlidingWindowAttentionMaskParamName);
    if (it == in_ports.end()) {
        return;
    }
    auto mask_tensor = request->get_tensor(it->second);
    fill_causal_sliding_mask(mask_tensor, num_stored_tokens_before, num_real_new_tokens, window_size);
    if (token_type_ids_real != nullptr) {
        overlay_vision_bidirectional_mask(mask_tensor, token_type_ids_real, num_real_new_tokens);
    }
}

void ov::npuw::util::overlay_vision_bidirectional_mask(ov::SoPtr<ov::ITensor> mask_tensor,
                                                       const int64_t* token_type_ids_real,
                                                       uint32_t num_real_new_tokens) {
    if (num_real_new_tokens == 0) {
        return;
    }
    OPENVINO_ASSERT(token_type_ids_real != nullptr,
                    "overlay_vision_bidirectional_mask: token_type_ids_real must not be null");

    const auto mask_view = get_mask_view(mask_tensor, num_real_new_tokens, "overlay_vision_bidirectional_mask");

    constexpr float kAttend = 0.0f;
    auto apply_vision_run = [&](uint32_t run_start, uint32_t run_end) {
        const uint32_t run_len = run_end - run_start;
        const uint32_t col_start = mask_view.past_width + mask_view.row_pad + run_start;
        for (uint32_t i = run_start; i < run_end; ++i) {
            float* row_ptr = mask_view.data + static_cast<size_t>(mask_view.row_pad + i) * mask_view.col_dim;
            std::fill_n(row_ptr + col_start, run_len, kAttend);
        }
    };

    uint32_t i = 0;
    while (i < num_real_new_tokens) {
        if (token_type_ids_real[i] != 1) {
            ++i;
            continue;
        }
        const uint32_t run_start = i;
        while (i < num_real_new_tokens && token_type_ids_real[i] == 1) {
            ++i;
        }
        apply_vision_run(run_start, i);
    }
}
