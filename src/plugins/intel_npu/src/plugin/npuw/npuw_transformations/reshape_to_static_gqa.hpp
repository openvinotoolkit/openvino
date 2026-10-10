// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_map>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Reshapes a GQA model's dynamic KV-cache (past_key/past_value) and attention-bias/mask
// Parameters to a static capacity of `max_seq_len` tokens, as required for NPU
// compilation. Intended to run on a private clone of the original (still-dynamic) model,
// since it mutates the model it's given in place -- the caller keeps the original around
// for tensor-shape bookkeeping.
class ReshapeToStaticGQA : public ov::pass::ModelPass {
    size_t m_max_seq_len;
    std::unordered_map<std::string, size_t> m_dynamic_kv_cache_axes;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::ReshapeToStaticGQA");
    explicit ReshapeToStaticGQA(size_t max_seq_len);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

    // The dynamic-axis Parameters (KV-cache + attention-bias) discovered and reshaped by
    // the last run_on_model() call, keyed by Parameter friendly name. Empty until
    // run_on_model() has been called.
    const std::unordered_map<std::string, size_t>& dynamic_kv_cache_axes() const {
        return m_dynamic_kv_cache_axes;
    }
};

}  // namespace ov::npuw
