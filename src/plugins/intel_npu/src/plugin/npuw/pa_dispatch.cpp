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
    const auto& bib = d.block_indices_begins;
    const auto n_seqs = d.sequences();
    const auto n_tokens = d.tokens();

    expect(d.input_ids_size == n_tokens, "input_ids size != subsequence_begins token count");
    expect(d.position_ids_token_count == n_tokens, "position_ids last dim != subsequence_begins token count");
    expect(static_cast<int64_t>(sub.size()) == n_seqs + 1, "subsequence_begins size != past_lens size + 1");
    expect(sub.front() == 0, "subsequence_begins does not start at 0");
    expect(std::is_sorted(sub.begin(), sub.end()) && std::adjacent_find(sub.begin(), sub.end()) == sub.end(),
           "subsequence_begins is not strictly increasing");
    expect(static_cast<int64_t>(bib.size()) == n_seqs + 1, "block_indices_begins size != past_lens size + 1");
    expect(bib.front() == 0 && bib.back() == static_cast<int64_t>(d.block_indices.size()) &&
               std::is_sorted(bib.begin(), bib.end()),
           "block_indices_begins is not a prefix-sum over block_indices");

    // Per-subsequence: the provided blocks must cover past + scheduled tokens,
    // and max_context_len bounds every context.
    for (int64_t s = 0; s < n_seqs; ++s) {
        const auto ctx_after = past[s] + (sub[s + 1] - sub[s]);
        expect(past[s] >= 0, "negative past_lens entry");
        expect(d.max_context_len >= ctx_after, "max_context_len < a subsequence's context length");
        expect((bib[s + 1] - bib[s]) * static_cast<int64_t>(block_size) >= ctx_after,
               "block_indices do not cover a subsequence's context");
    }

    // Gather contract: sampled_tokens_indices picks which flat token rows get
    // logits; an empty selection is legal (intermediate prefill chunks).
    for (auto idx : d.sampled_tokens_indices) {
        expect(idx >= 0 && idx < n_tokens, "sampled_tokens_indices out of token range");
    }
}

int64_t ov::npuw::pa::Chunk::tokens() const {
    int64_t n = 0;
    for (const auto& p : pieces) {
        n += p.tokens;
    }
    return n;
}

std::vector<ov::npuw::pa::Chunk> ov::npuw::pa::plan_dispatch(const Dispatch& d,
                                                             const std::vector<std::size_t>& variant_token_dims,
                                                             std::size_t max_sampled) {
    OPENVINO_ASSERT(!variant_token_dims.empty() && max_sampled > 0, "PA dispatch: no variants to plan over");
    std::vector<int64_t> dims(variant_token_dims.begin(), variant_token_dims.end());
    std::sort(dims.begin(), dims.end());
    dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
    OPENVINO_ASSERT(dims.front() == 1, "PA dispatch: the 1-token variant is required");
    const auto largest = dims.back();

    // Padding into the smallest multi-token variant is always cheap enough;
    // beyond it, a variant may hold at most four times the real tokens. One
    // infer with padding beats several small ones: the per-infer cost of
    // the trunk (its weights) dominates at these sizes.
    const auto smallest_multi = dims.size() > 1 ? dims[1] : dims[0];
    constexpr int64_t kMaxPadFactor = 4;

    // The smallest variant that holds n tokens (n <= largest).
    const auto fit = [&](int64_t n) {
        return *std::lower_bound(dims.begin(), dims.end(), n);
    };
    // Sampled rows falling into tokens [offset, offset + n) of subsequence seq.
    const auto sampled_in = [&](int64_t seq, int64_t offset, int64_t n) {
        const auto g0 = d.subsequence_begins[seq] + offset;
        return std::count_if(d.sampled_tokens_indices.begin(), d.sampled_tokens_indices.end(), [&](int64_t g) {
            return g >= g0 && g < g0 + n;
        });
    };

    std::vector<Chunk> chunks;
    Chunk singles;  // single-token subsequences, at most one sampled row each
    const auto flush_singles = [&]() {
        if (!singles.pieces.empty()) {
            singles.token_dim = static_cast<std::size_t>(fit(singles.tokens()));
            chunks.push_back(std::move(singles));
            singles = Chunk{};
        }
    };

    for (int64_t seq = 0; seq < d.sequences(); ++seq) {
        const auto len = d.subsequence_begins[seq + 1] - d.subsequence_begins[seq];
        if (len == 1) {
            if (singles.pieces.size() == max_sampled) {
                flush_singles();
            }
            singles.pieces.push_back(Piece{seq, 0, 1});
            continue;
        }
        for (int64_t offset = 0; offset < len;) {
            const auto remaining = len - offset;
            int64_t n = 0, dim = 0;
            if (remaining >= largest) {
                n = dim = largest;
            } else if (const auto padded = fit(remaining);
                       padded <= std::max(kMaxPadFactor * remaining, smallest_multi)) {
                n = remaining;
                dim = padded;
            } else {
                // Too much padding: take the largest variant that fits and
                // keep going.
                n = dim = *std::prev(std::upper_bound(dims.begin(), dims.end(), remaining));
            }
            OPENVINO_ASSERT(sampled_in(seq, offset, n) <= static_cast<int64_t>(max_sampled),
                            "PA dispatch: a chunk of subsequence ",
                            seq,
                            " samples more than ",
                            max_sampled,
                            " tokens");
            chunks.push_back(Chunk{static_cast<std::size_t>(dim), {Piece{seq, offset, n}}});
            offset += n;
        }
    }
    flush_singles();
    return chunks;
}

std::string ov::npuw::pa::to_string(const std::vector<Chunk>& chunks) {
    std::string out;
    for (const auto& c : chunks) {
        out += (out.empty() ? "" : " ") + std::to_string(c.tokens()) + "/" + std::to_string(c.token_dim) + "[";
        for (const auto& p : c.pieces) {
            out += (&p == &c.pieces.front() ? "" : ",") + std::to_string(p.seq) + "+" + std::to_string(p.offset) + ":" +
                   std::to_string(p.tokens);
        }
        out += "]";
    }
    return out;
}
