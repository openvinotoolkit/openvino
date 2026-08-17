// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ov::npuw::pa {

// One dispatch's control tensors, parsed out of the infer request. The
// vectors are plain copies of the (small, integer) PA control inputs, which
// keeps the validation below a pure function over host data.
struct Dispatch {
    std::vector<int64_t> past_lens;
    std::vector<int64_t> subsequence_begins;
    std::vector<int64_t> block_indices;           // meaningful when has_block_table
    std::vector<int64_t> block_indices_begins;    // meaningful when has_block_table
    std::vector<int64_t> sampled_tokens_indices;  // meaningful when has_sampled_tokens
    int64_t max_context_len = 0;
    int64_t input_ids_size = -1;           // -1 when the model has no input_ids input
    int64_t position_ids_token_count = 0;  // last dim of position_ids
    bool has_block_table = false;
    bool has_sampled_tokens = false;

    // subsequence_begins is the source of truth for the flat token dimension.
    int64_t tokens() const {
        return subsequence_begins.empty() ? int64_t{0} : subsequence_begins.back();
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

}  // namespace ov::npuw::pa
