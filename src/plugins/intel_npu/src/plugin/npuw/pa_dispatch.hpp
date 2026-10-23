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

// True when the dispatch runs chunked over the variants: some subsequence fills
// a multi-token variant, or it is a single-sequence decode. A decode batch or a
// short prefill stays one 1:1 infer, which chunking would only serialize.
bool variants_serve(const Dispatch& dispatch, const std::vector<std::size_t>& chunk_sizes);

// One infer: rows [begin, begin + tokens) of the flat token axis. chunk_size is
// the variant that runs them, or 0 for the dynamic model.
struct Chunk {
    int64_t begin = 0;
    int64_t tokens = 0;
    std::size_t chunk_size = 0;

    bool dynamic() const {
        return chunk_size == 0u;
    }
};

// Splits the flat token axis into chunks, regardless of subsequence
// boundaries: the largest chunk size that fits the rows left, the 1-token one
// only for a single row, otherwise the dynamic model takes all the rows left.
std::vector<Chunk> plan_chunks(const Dispatch& dispatch, const std::vector<std::size_t>& chunk_sizes);

// The controls of one chunk. Every subsequence the chunk cuts becomes one
// subsequence of the chunk, rebased past its rows sent before and with its
// full block table. The chunk's subsequences are first_sequence onwards, in
// order. sampled_tokens_indices are local to the chunk, and logits_rows gives
// the row of each in the caller's logits.
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
