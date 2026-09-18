// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ov::npuw::pa {

// One dispatch's control tensors, parsed out of the infer request. The
// vectors are plain copies of the (small, integer) PA control inputs, which
// keeps the validation and planning below pure functions over host data.
struct Dispatch {
    std::vector<int64_t> past_lens;
    std::vector<int64_t> subsequence_begins;
    std::vector<int64_t> block_indices;
    std::vector<int64_t> block_indices_begins;
    std::vector<int64_t> sampled_tokens_indices;
    int64_t max_context_len = 0;
    int64_t input_ids_size = 0;
    int64_t position_ids_token_count = 0;  // last dim of position_ids

    // subsequence_begins is the source of truth for the flat token dimension.
    int64_t tokens() const {
        return subsequence_begins.empty() ? int64_t{0} : subsequence_begins.back();
    }
    int64_t sequences() const {
        return static_cast<int64_t>(past_lens.size());
    }
};

// Validates one dispatch against the PA model expectations; throws with a
// specific message on the first violation. block_size is the KV cache block
// size the block tables are checked against. dispatch_idx only labels the
// error message.
void validate_dispatch(const Dispatch& dispatch, std::size_t block_size, std::size_t dispatch_idx);

// A run of scheduled tokens of one subsequence: tokens [offset, offset +
// tokens) of subsequence seq, in the caller's numbering.
struct Piece {
    int64_t seq = 0;
    int64_t offset = 0;
    int64_t tokens = 0;
};

// One variant infer: the pieces packed, in this order, into the variant
// with token_dim tokens. The pieces never exceed token_dim; the rest of the
// variant's rows is padding.
struct Chunk {
    std::size_t token_dim = 0;
    std::vector<Piece> pieces;

    int64_t tokens() const;
};

// Splits a dispatch into variant infers. variant_token_dims are the fixed
// token sizes available, in any order; the 1-token variant is required. A
// subsequence is consumed largest variant first; what is left takes the
// smallest variant that fits it when that is the smallest multi-token
// variant or holds at most four times the real tokens, and is split further
// otherwise. Single-token subsequences (a decode batch) are packed together,
// max_sampled per chunk, so a batch stays one infer. Pieces of one
// subsequence come out in order. Throws when a chunk would produce more than
// max_sampled logits rows.
std::vector<Chunk> plan_dispatch(const Dispatch& dispatch,
                                 const std::vector<std::size_t>& variant_token_dims,
                                 std::size_t max_sampled);

// "tokens/variant[seq+offset:tokens,...] ..." for logs and tests.
std::string to_string(const std::vector<Chunk>& chunks);

}  // namespace ov::npuw::pa
