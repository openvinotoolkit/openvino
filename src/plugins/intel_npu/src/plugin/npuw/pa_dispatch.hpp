// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ov::npuw::pa {

// One dispatch's control tensors, parsed out of the infer request. The
// vectors are plain copies of the (small) PA control inputs in their own
// element types, which keeps the validation below a pure function over host
// data: the PA op takes its controls as i32, the lm_head gather index is i64.
struct Dispatch {
    std::vector<int32_t> past_lens;
    std::vector<int32_t> subsequence_begins;
    std::vector<int32_t> block_indices;           // meaningful when has_block_table
    std::vector<int32_t> block_indices_begins;    // meaningful when has_block_table
    std::vector<int64_t> sampled_tokens_indices;  // meaningful when has_sampled_tokens
    int64_t max_context_len = 0;
    int64_t input_ids_size = -1;           // -1 when the model has no input_ids input
    int64_t position_ids_token_count = 0;  // last dim of position_ids
    bool has_block_table = false;
    bool has_sampled_tokens = false;

    // subsequence_begins is the source of truth for the flat token dimension.
    int64_t tokens() const {
        return subsequence_begins.empty() ? int64_t{0} : int64_t{subsequence_begins.back()};
    }
    int64_t sequences() const {
        return static_cast<int64_t>(past_lens.size());
    }
};

// Validates one dispatch against the PA model expectations; throws with a
// specific message on the first violation. A block_size of 0 means the cache
// block geometry is still dynamic, and block-table coverage is not checked.
// dispatch_idx only labels the error message.
void validate_dispatch(const Dispatch& dispatch, std::size_t block_size, std::size_t dispatch_idx);

// Decides whether a dispatch runs on the pre-compiled semi-static variants
// (chunked, per subsequence) or 1:1 on the dynamic model, given the variants'
// fixed token sizes. The dynamic model handles the whole flat dispatch in a
// single infer, so chunking must earn its decomposition: a dispatch qualifies
// when some subsequence can fill a multi-token variant, or when it is a
// single-sequence decode - the 1-token variant's own case, one infer either
// way. A decode batch and a short prefill run 1:1, where per-subsequence
// calls would only serialize them.
bool variants_serve(const Dispatch& dispatch, const std::vector<std::size_t>& variant_token_dims);

}  // namespace ov::npuw::pa
