// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_dispatch.hpp"

#include <algorithm>
#include <functional>

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

std::vector<ov::npuw::pa::Chunk> ov::npuw::pa::plan_chunks(const Dispatch& d,
                                                           const std::vector<std::size_t>& chunk_sizes) {
    OPENVINO_ASSERT(!chunk_sizes.empty(), "PA chunk plan: no chunk sizes");
    std::vector<std::size_t> sizes(chunk_sizes);
    std::sort(sizes.begin(), sizes.end(), std::greater<std::size_t>());

    std::vector<Chunk> chunks;
    for (int64_t begin = 0; begin < d.tokens();) {
        const auto rows = d.tokens() - begin;
        const auto fits = std::find_if(sizes.begin(), sizes.end(), [&](std::size_t s) {
            return static_cast<int64_t>(s) <= rows && (s > 1u || rows == 1);
        });
        if (fits != sizes.end()) {
            chunks.push_back(Chunk{begin, static_cast<int64_t>(*fits), *fits});
        } else {
            chunks.push_back(Chunk{begin, rows, 0u});
        }
        begin += chunks.back().tokens;
    }
    return chunks;
}

ov::npuw::pa::ChunkControls ov::npuw::pa::chunk_controls(const Dispatch& d, const Chunk& chunk) {
    const auto begin = chunk.begin;
    const auto end = chunk.begin + chunk.tokens;

    ChunkControls c;
    c.first_sequence = -1;
    c.subsequence_begins.push_back(0);
    c.block_indices_begins.push_back(0);
    for (int64_t s = 0; s < d.sequences(); ++s) {
        const int64_t lo = std::max<int64_t>(d.subsequence_begins[s], begin);
        const int64_t hi = std::min<int64_t>(d.subsequence_begins[s + 1], end);
        if (lo >= hi) {
            continue;
        }
        if (c.first_sequence < 0) {
            c.first_sequence = s;
        }
        c.past_lens.push_back(static_cast<int32_t>(d.past_lens[s] + (lo - d.subsequence_begins[s])));
        c.subsequence_begins.push_back(static_cast<int32_t>(hi - begin));
        if (d.has_block_table) {
            const auto blocks_begin = d.block_indices.begin() + d.block_indices_begins[s];
            const auto blocks_end = d.block_indices.begin() + d.block_indices_begins[s + 1];
            c.block_indices.insert(c.block_indices.end(), blocks_begin, blocks_end);
            c.block_indices_begins.push_back(static_cast<int32_t>(c.block_indices.size()));
        }
    }
    OPENVINO_ASSERT(c.first_sequence >= 0, "PA chunk [", begin, ", ", end, ") holds no rows of the dispatch");

    for (std::size_t i = 0; i < d.sampled_tokens_indices.size(); ++i) {
        const auto row = d.sampled_tokens_indices[i];
        if (row >= begin && row < end) {
            c.sampled_tokens_indices.push_back(row - begin);
            c.logits_rows.push_back(i);
        }
    }
    return c;
}
