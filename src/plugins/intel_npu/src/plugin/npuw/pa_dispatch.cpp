// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_dispatch.hpp"

#include <algorithm>

#include "openvino/core/except.hpp"

void ov::npuw::pa::validate_dispatch(const Dispatch& d, std::size_t block_size, std::size_t dispatch_idx) {
    const auto expect = [&](bool cond, const char* what) {
        OPENVINO_ASSERT(cond, "PA dispatch #", dispatch_idx, " violates the PA model expectations: ", what);
    };

    const auto& past = d.past_lens;
    const auto& sub = d.subsequence_begins;
    const auto n_seqs = d.sequences();
    const auto n_tokens = d.tokens();

    // Embedding models have no input_ids; M-RoPE position_ids carry the token
    // count in the last dim.
    if (d.input_ids_size >= 0) {
        expect(d.input_ids_size == n_tokens, "input_ids size != subsequence_begins token count");
    }
    expect(d.position_ids_token_count == n_tokens, "position_ids last dim != subsequence_begins token count");
    expect(static_cast<int64_t>(sub.size()) == n_seqs + 1, "subsequence_begins size != past_lens size + 1");
    expect(sub.front() == 0, "subsequence_begins does not start at 0");
    expect(std::is_sorted(sub.begin(), sub.end()) && std::adjacent_find(sub.begin(), sub.end()) == sub.end(),
           "subsequence_begins is not strictly increasing");

    // Eviction models carry per-layer block tables instead.
    if (d.has_block_table) {
        const auto& bib = d.block_indices_begins;
        expect(static_cast<int64_t>(bib.size()) == n_seqs + 1, "block_indices_begins size != past_lens size + 1");
        expect(bib.front() == 0 && bib.back() == static_cast<int64_t>(d.block_indices.size()) &&
                   std::is_sorted(bib.begin(), bib.end()),
               "block_indices_begins is not a prefix-sum over block_indices");
    }

    for (int64_t s = 0; s < n_seqs; ++s) {
        const auto ctx_after = past[s] + (sub[s + 1] - sub[s]);
        expect(past[s] >= 0, "negative past_lens entry");
        expect(d.max_context_len >= ctx_after, "max_context_len < a subsequence's context length");
        if (d.has_block_table && block_size > 0) {
            expect((d.block_indices_begins[s + 1] - d.block_indices_begins[s]) * static_cast<int64_t>(block_size) >=
                       ctx_after,
                   "block_indices do not cover a subsequence's context");
        }
    }

    // An empty selection is legal (intermediate prefill chunks).
    if (d.has_sampled_tokens) {
        for (auto idx : d.sampled_tokens_indices) {
            expect(idx >= 0 && idx < n_tokens, "sampled_tokens_indices out of token range");
        }
    }
}

bool ov::npuw::pa::variants_serve(const Dispatch& dispatch, const std::vector<std::size_t>& chunk_sizes) {
    if (chunk_sizes.empty() || dispatch.tokens() == 0) {
        return false;
    }

    bool has_one_token_variant = false;
    std::size_t min_multi_token = 0u;
    for (const auto chunk_size : chunk_sizes) {
        has_one_token_variant |= (chunk_size == 1u);
        if (chunk_size > 1u && (min_multi_token == 0u || chunk_size < min_multi_token)) {
            min_multi_token = chunk_size;
        }
    }

    if (dispatch.sequences() == 1 && dispatch.tokens() == 1) {
        return has_one_token_variant;
    }

    for (int64_t s = 0; min_multi_token > 0u && s < dispatch.sequences(); ++s) {
        const auto seq_len = dispatch.subsequence_begins[s + 1] - dispatch.subsequence_begins[s];
        if (seq_len >= static_cast<int64_t>(min_multi_token)) {
            return true;
        }
    }
    return false;
}
