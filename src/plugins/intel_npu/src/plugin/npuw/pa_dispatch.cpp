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

std::vector<ov::npuw::pa::Call> ov::npuw::pa::plan_calls(const Dispatch& dispatch,
                                                         const std::vector<std::size_t>& chunk_sizes) {
    std::vector<std::size_t> multi_token;
    bool has_one_token_variant = false;
    for (const auto chunk_size : chunk_sizes) {
        has_one_token_variant |= (chunk_size == 1u);
        if (chunk_size > 1u) {
            multi_token.push_back(chunk_size);
        }
    }
    std::sort(multi_token.begin(), multi_token.end(), std::greater<std::size_t>());
    const auto largest_fit = [&](int64_t n) {
        for (const auto chunk_size : multi_token) {
            if (static_cast<int64_t>(chunk_size) <= n) {
                return chunk_size;
            }
        }
        return std::size_t{0u};
    };

    std::vector<Call> calls;
    std::vector<Segment> singles;
    std::vector<Segment> rest;
    for (int64_t s = 0; s < dispatch.sequences(); ++s) {
        const int64_t seq_len = dispatch.subsequence_begins[s + 1] - dispatch.subsequence_begins[s];
        int64_t off = 0;
        while (const auto chunk_size = largest_fit(seq_len - off)) {
            const auto n = static_cast<int64_t>(chunk_size);
            calls.push_back({chunk_size, {{s, off, n}}});
            off += n;
        }
        if (seq_len - off == 1) {
            singles.push_back({s, off, 1});
        } else if (seq_len > off) {
            rest.push_back({s, off, seq_len - off});
        }
    }

    auto next = singles.begin();
    while (const auto chunk_size = largest_fit(singles.end() - next)) {
        const auto end = next + static_cast<std::ptrdiff_t>(chunk_size);
        calls.push_back({chunk_size, std::vector<Segment>(next, end)});
        next = end;
    }
    if (singles.end() - next == 1 && has_one_token_variant) {
        calls.push_back({1u, {*next}});
        ++next;
    }

    if (calls.empty()) {
        return {};
    }
    rest.insert(rest.end(), next, singles.end());
    if (!rest.empty()) {
        std::sort(rest.begin(), rest.end(), [](const Segment& a, const Segment& b) {
            return a.seq < b.seq;
        });
        calls.push_back({0u, std::move(rest)});
    }
    return calls;
}
