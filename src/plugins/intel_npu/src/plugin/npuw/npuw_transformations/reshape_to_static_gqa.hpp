// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <unordered_map>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Reshapes a GQA model's dynamic KV-cache (past_key/past_value) and attention-bias/mask
// Parameters to a static capacity, as required for NPU compilation. Most layers get the
// model-wide `max_seq_len`; a sliding-window layer instead gets its own, smaller capacity,
// resolved via whichever of these mechanisms a model uses to declare it, in priority order
// (see the .cpp for details):
//  1. rt_info["framework"]["sliding_window_layers"/"sliding_window_size"] (derived from
//     ONNX metadata_props -- this app's actual models' authoritative mechanism);
//  2. a GQA op's own sliding_window_cache=true + local_window_size attributes (the
//     ONNX-standard, in-graph ring-buffer eviction case);
//  3. if neither is present and the caller configured `default_sliding_window_capacity`, a
//     best-effort fallback that detects sliding-window layers purely from a shared
//     SEQLENS_K/TOTAL_SEQUENCE_LENGTH input name hint and pins them to that caller-supplied
//     default (never a value read from the model itself).
// Intended to run on a private clone of the original (still-dynamic) model, since it
// mutates the model it's given in place -- the caller keeps the original around for
// tensor-shape bookkeeping.
class ReshapeToStaticGQA : public ov::pass::ModelPass {
    size_t m_max_seq_len;
    std::optional<size_t> m_default_sliding_window_capacity;
    std::unordered_map<std::string, size_t> m_dynamic_kv_cache_axes;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::ReshapeToStaticGQA");
    explicit ReshapeToStaticGQA(size_t max_seq_len,
                                std::optional<size_t> default_sliding_window_capacity = std::nullopt);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

    // The dynamic-axis Parameters (KV-cache + attention-bias) discovered and reshaped by
    // the last run_on_model() call, keyed by Parameter friendly name. Empty until
    // run_on_model() has been called.
    const std::unordered_map<std::string, size_t>& dynamic_kv_cache_axes() const {
        return m_dynamic_kv_cache_axes;
    }
};

}  // namespace ov::npuw
