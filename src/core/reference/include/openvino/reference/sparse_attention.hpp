// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov::reference {

/// Reference for v17::SparseAttentionIndexer with one selection head.
/// q [N, Hi, L, Di], keys [N, B, Di], key_length[N or 1], weights [N, L, Hi] or nullptr (unit weights).
/// Query token l of sequence n sits at position p = key_length[n] - L + l and sees the complete blocks
/// j < (p + 1) / key_block_size (and j < B). score_j = sum_h w[h] * max(0, q[h] . key_j); the best
/// min(visible, token_budget / key_block_size) blocks are written in descending score order (ties: lower
/// index first); unused slots hold -1. indices [N, 1, L, KB], count [N, 1, L].
template <typename T>
void sparse_attention_indexer(const T* q,
                              const T* keys,
                              const int64_t* key_length,
                              size_t key_length_size,
                              const T* weights,
                              int32_t* indices,
                              int32_t* count,
                              size_t N,
                              size_t Hi,
                              size_t L,
                              size_t Di,
                              size_t B,
                              size_t key_block_size,
                              size_t KB) {
    std::vector<float> score(B);
    std::vector<size_t> order(B);
    for (size_t n = 0; n < N; n++) {
        const int64_t K = key_length[key_length_size == 1 ? 0 : n];
        for (size_t l = 0; l < L; l++) {
            const int64_t p = K - static_cast<int64_t>(L) + static_cast<int64_t>(l);
            const size_t visible =
                p < 0 ? 0 : std::min(B, static_cast<size_t>((p + 1) / static_cast<int64_t>(key_block_size)));
            for (size_t j = 0; j < visible; j++) {
                float s = 0.0f;
                for (size_t h = 0; h < Hi; h++) {
                    float dot = 0.0f;
                    const T* qh = q + ((n * Hi + h) * L + l) * Di;
                    const T* kj = keys + (n * B + j) * Di;
                    for (size_t d = 0; d < Di; d++)
                        dot += static_cast<float>(qh[d]) * static_cast<float>(kj[d]);
                    const float w = weights ? static_cast<float>(weights[(n * L + l) * Hi + h]) : 1.0f;
                    s += w * std::max(0.0f, dot);
                }
                score[j] = s;
            }
            std::iota(order.begin(), order.begin() + visible, size_t{0});
            std::stable_sort(order.begin(), order.begin() + visible, [&](size_t a, size_t b) {
                return score[a] > score[b];
            });
            const size_t take = std::min(visible, KB);
            int32_t* row = indices + (n * L + l) * KB;
            for (size_t k = 0; k < KB; k++)
                row[k] = k < take ? static_cast<int32_t>(order[k]) : -1;
            count[n * L + l] = static_cast<int32_t>(take);
        }
    }
}

/// Reference for v17::SparseAttention. q [N, H, L, E], k [N, Hkv, S, E], v [N, Hkv, S, Ev],
/// block_indices [N, Hs, Lq, KB], block_count [N, Hs, Lq]; key_length[N or 1] (nullptr: S);
/// scale (nullptr: 1/sqrt(E)); sinks [H] or nullptr. Query l of sequence n is at p = K - L + l and uses
/// selection row l / query_block_size of selection head h * Hs / H; it attends the tokens of its selected
/// blocks that are <= p and < K and, when causal, its incomplete block [((p + 1) / R) * R, p]. Positions are
/// logical; batches are right-aligned (left padding), so logical position c of sequence n is physical key row
/// pad + c with pad = S - K (the opset-17 "Batches with padding" rule).
template <typename T>
void sparse_attention(const T* q,
                      const T* k,
                      const T* v,
                      const int32_t* block_indices,
                      const int32_t* block_count,
                      const int64_t* key_length,
                      size_t key_length_size,
                      const T* scale,
                      const T* sinks,
                      T* out,
                      size_t N,
                      size_t H,
                      size_t Hkv,
                      size_t L,
                      size_t S,
                      size_t E,
                      size_t Ev,
                      size_t Hs,
                      size_t Lq,
                      size_t KB,
                      size_t key_block_size,
                      size_t query_block_size,
                      bool causal) {
    const float sc = scale ? static_cast<float>(scale[0]) : 1.0f / std::sqrt(static_cast<float>(E));
    const size_t R = key_block_size;
    std::vector<size_t> cells;
    std::vector<float> logits;
    std::vector<char> used(S);
    for (size_t n = 0; n < N; n++) {
        const int64_t K = key_length ? key_length[key_length_size == 1 ? 0 : n] : static_cast<int64_t>(S);
        OPENVINO_ASSERT(K >= 0 && K <= static_cast<int64_t>(S), "SparseAttention: key_length must be in [0, S]");
        const size_t pad = S - static_cast<size_t>(K);
        for (size_t h = 0; h < H; h++) {
            const size_t hkv = h / (H / Hkv);
            const size_t hs = h * Hs / H;
            for (size_t l = 0; l < L; l++) {
                const int64_t p = K - static_cast<int64_t>(L) + static_cast<int64_t>(l);
                const int64_t limit = std::min<int64_t>(p, K - 1);
                const size_t lq = l / query_block_size;
                cells.clear();
                std::fill(used.begin(), used.end(), 0);
                auto add = [&](int64_t c) {
                    if (c >= 0 && c <= limit && c < static_cast<int64_t>(S) && !used[c]) {
                        used[c] = 1;
                        cells.push_back(static_cast<size_t>(c));
                    }
                };
                const size_t sel = (n * Hs + hs) * Lq + lq;
                for (int32_t i = 0; i < block_count[sel]; i++) {
                    const int64_t b = block_indices[sel * KB + i];
                    for (size_t t = 0; t < R; t++)
                        add(b * static_cast<int64_t>(R) + static_cast<int64_t>(t));
                }
                if (causal && p >= 0)
                    for (int64_t c = (p + 1) / static_cast<int64_t>(R) * static_cast<int64_t>(R); c <= p; c++)
                        add(c);
                T* o = out + ((n * H + h) * L + l) * Ev;
                const T* qv = q + ((n * H + h) * L + l) * E;
                float mx = sinks ? static_cast<float>(sinks[h]) : -std::numeric_limits<float>::infinity();
                logits.resize(cells.size());
                for (size_t i = 0; i < cells.size(); i++) {
                    const T* kv = k + ((n * Hkv + hkv) * S + pad + cells[i]) * E;
                    float dot = 0.0f;
                    for (size_t d = 0; d < E; d++)
                        dot += static_cast<float>(qv[d]) * static_cast<float>(kv[d]);
                    logits[i] = dot * sc;
                    mx = std::max(mx, logits[i]);
                }
                float den = sinks ? std::exp(static_cast<float>(sinks[h]) - mx) : 0.0f;
                for (auto& x : logits) {
                    x = std::exp(x - mx);
                    den += x;
                }
                for (size_t d = 0; d < Ev; d++) {
                    float acc = 0.0f;
                    for (size_t i = 0; i < cells.size(); i++)
                        acc += logits[i] * static_cast<float>(v[((n * Hkv + hkv) * S + pad + cells[i]) * Ev + d]);
                    o[d] = static_cast<T>(den > 0.0f ? acc / den : 0.0f);
                }
            }
        }
    }
}

}  // namespace ov::reference
