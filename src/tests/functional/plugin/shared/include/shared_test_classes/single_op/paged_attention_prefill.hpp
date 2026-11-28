// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace test {
namespace paged_attention_prefill {

constexpr size_t MAX_CONTEXT_LEN = 1024;

// Single-op PagedAttention model with MHA (num_kv_heads == num_heads) for prefill tests against the TEMPLATE reference.
// Only q, k, v, caches, past_lens, subsequence_begins, block_indices(_begins) and the enabled optional inputs are
// parameters; all other inputs are disabled constants.
struct ModelOptions {
    ov::element::Type data_type;
    int64_t head_size;
    int64_t head_num;
    int32_t sliding_window;
    bool cpu_layout = false;  //< CPU cache layout and block size; GPU otherwise.
    size_t batch_size = 1;
    bool token_type_ids = false;
    int64_t relative_bias_distances = 0;  //< 0 disables the relative bias.
};

std::shared_ptr<ov::Model> make_model(const ModelOptions& options);

// Inputs for batch_size sequences of seq_len new tokens with empty history.
std::map<std::string, ov::Tensor> make_prefill_inputs(const ModelOptions& options, size_t seq_len);

}  // namespace paged_attention_prefill
}  // namespace test
}  // namespace ov
