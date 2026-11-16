// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ov::npuw::pa {

// Host copies of one dispatch's control tensors, so the checks below are pure
// functions.
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

    int64_t tokens() const {
        return subsequence_begins.empty() ? int64_t{0} : int64_t{subsequence_begins.back()};
    }
    int64_t sequences() const {
        return static_cast<int64_t>(past_lens.size());
    }
};

// Throws on the first violation of the PA model expectations. A block_size of
// 0 skips the block-table coverage check.
void validate_dispatch(const Dispatch& dispatch, std::size_t block_size, std::size_t dispatch_idx);

// One model call: rows [begin, begin + tokens) of the flat token axis, run on
// the chunk_size variant, or on the dynamic model when chunk_size is 0.
struct Chunk {
    int64_t begin = 0;
    int64_t tokens = 0;
    std::size_t chunk_size = 0;

    bool dynamic() const {
        return chunk_size == 0u;
    }
};

// Cuts the flat token axis, ignoring subsequence boundaries: the largest size
// that fits the rows left (size 1 only for a single row), else one dynamic
// chunk for the rest.
std::vector<Chunk> plan_chunks(const Dispatch& dispatch, const std::vector<std::size_t>& chunk_sizes);

// The controls of one chunk: each subsequence it touches, from first_sequence
// on, with past_lens grown by its rows sent before and its full block table.
// sampled_tokens_indices are chunk-local; logits_rows maps each to the caller's
// logits row. Needs the block table (has_block_table).
struct ChunkControls {
    std::vector<int32_t> past_lens;
    std::vector<int32_t> subsequence_begins;
    std::vector<int32_t> block_indices;
    std::vector<int32_t> block_indices_begins;
    std::vector<int64_t> sampled_tokens_indices;
    std::vector<std::size_t> logits_rows;
    int64_t first_sequence = 0;
};

ChunkControls chunk_controls(const Dispatch& dispatch, const Chunk& chunk);

}  // namespace ov::npuw::pa
