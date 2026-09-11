// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the CM qsa primitive (design section 12).
//
// Covered:
//  - Q0: the main K/V cache update follows the PA page/slot contract;
//  - Q1: indexer Q RMSNorm + partial RoPE, raw-K stored unnormalized/unrotated,
//        FP32 mean pooling of complete blocks with block-start RoPE;
//  - Q2: ReLU before the head reduction, 1/sqrt(Di) after the sum, block-causal
//        condition p_b + r - 1 <= i, and top-k width min(block_topk, n_complete);
//  - Q3: selected blocks plus the always-visible tail, shared selection across all
//        query heads, and the optional attention output gate;
//  - prefill and decode produce consistent results;
//  - attribute validation rejects illegal configurations instead of truncating.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/paged_attention.hpp>
#include <intel_gpu/primitives/qsa.hpp>
#include <intel_gpu/runtime/profiling.hpp>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

#include "impls/cm/qsa/qsa_stage_events.hpp"
#include "qsa_inst.h"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

using QSAIn = ov::intel_gpu::op::QSA::InputIdx;

bool cm_qsa_available() {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    return cldnn::check_cm_jit_support(engine, config) && engine.get_device_info().supports_immad;
}

struct qsa_dims {
    size_t Hq = 2;
    size_t Hkv = 1;
    size_t Dh = 16;
    size_t Dv = 16;
    size_t D = 32;
    size_t Hidx = 2;
    size_t Di = 16;
    size_t rotary = 8;
    size_t r = 4;
    size_t budget = 8;
    size_t block_topk = 2;
    size_t pa_block = 16;
    size_t max_positions = 64;
    float eps = 1e-6f;
};

/// Shape that satisfies every constraint of the systolic prefill scoring kernel
/// (Di % 16 == 0, Di * 2 >= 64 bytes of 2D-block surface width, and
/// (pa_block_size / r) % 8 == 0 so an 8-block A tile stays inside one PA page),
/// so this configuration exercises qsa_score_topk_dpas while the default `qsa_dims`
/// exercises the scalar qsa_score_topk_fused fallback. Both must agree with the
/// same FP32 reference.
qsa_dims dpas_dims() {
    qsa_dims d;
    d.Hq = 2;
    d.Hkv = 1;
    d.Dh = 32;
    d.Dv = 32;
    d.D = 64;
    d.Hidx = 2;
    d.Di = 64;
    d.rotary = 16;
    d.r = 4;
    d.budget = 16;
    d.block_topk = 4;
    d.pa_block = 64;
    d.max_positions = 128;
    return d;
}

/// The shape the production Qwen3.8 checkpoint actually uses, scaled down only in the
/// dimensions that cost test time (heads and budget). Everything that changes a code
/// path is kept at its real value: head_dim = 256 (the register-pressure case that
/// forces the head-dim worker split), Hkv > 1 (non-trivial KV-cache head stride),
/// indexer_head_dim = 128 with rotary_dim = 64 (rotary/Di = 1/2), and the real
/// PagedAttention page size of 16 (so summaries_per_page = 4, not a multiple of the
/// 8-row dpas A tile).
qsa_dims prod_dims(size_t max_positions = 128) {
    qsa_dims d;
    d.Hq = 4;
    d.Hkv = 2;
    d.Dh = 256;
    d.Dv = 256;
    d.D = 256;
    d.Hidx = 4;
    d.Di = 128;
    d.rotary = 64;
    d.r = 4;
    d.budget = 16;
    d.block_topk = 4;
    d.pa_block = cldnn::paged_attention::block_size;
    d.max_positions = max_positions;
    return d;
}

/// The production head counts. `prod_dims` keeps Hq small to bound the reference cost, but
/// that leaves n_rep = Hq / Hkv = 2, whereas the checkpoint has 24 / 2 = 12. n_rep decides
/// how many heads the systolic attention can pack into one dpas tile (they must share a KV
/// head), so it changes the tiling: n_rep = 2 gives a 2-head tile, n_rep = 12 a 4-head one.
qsa_dims prod_gqa_dims(size_t max_positions = 128) {
    auto d = prod_dims(max_positions);
    d.Hq = 24;
    d.Hkv = 2;
    return d;
}

/// The indexer is only exercised end-to-end when the top-k actually discards blocks: if
/// `block_topk >= n_complete` every visible block is selected and the whole selection
/// pipeline (projection, RMSNorm, RoPE, scoring, top-k) becomes a no-op that no output
/// comparison can detect. Every test that means to cover the indexer calls this first.
#define ASSERT_SELECTION_IS_DISCRIMINATING(d, past_len, q_len)                                          \
    do {                                                                                                \
        const size_t n_complete_ = ((past_len) + (q_len)) / (d).r;                                       \
        ASSERT_GE(n_complete_, 3 * (d).block_topk)                                                       \
            << "the top-k selects " << (d).block_topk << " of only " << n_complete_                      \
            << " blocks, so a wrong selection cannot change the output";                                 \
    } while (0)

ov::intel_gpu::op::QSA::Config make_config(const qsa_dims& d, bool has_gate) {
    ov::intel_gpu::op::QSA::Config c;
    c.heads_num = d.Hq;
    c.kv_heads_num = d.Hkv;
    c.k_head_size = d.Dh;
    c.v_head_size = d.Dv;
    c.hidden_size = d.D;
    c.indexer_heads = d.Hidx;
    c.indexer_kv_heads = 1;
    c.indexer_head_dim = d.Di;
    c.indexer_rotary_dim = d.rotary;
    c.budget = d.budget;
    c.compress_ratio = d.r;
    c.qsa_block_tokens = d.r;
    c.pa_block_size = d.pa_block;
    c.block_topk = d.block_topk;
    c.scale = 1.0f / std::sqrt(static_cast<float>(d.Dh));
    c.indexer_norm_eps = d.eps;
    c.has_output_gate = has_gate;
    c.is_causal = true;
    return c;
}

std::vector<ov::float16> to_f16(const std::vector<float>& src) {
    std::vector<ov::float16> dst(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst[i] = ov::float16(src[i]);
    }
    return dst;
}

memory::ptr make_f16(engine& eng, const ov::PartialShape& shape, const std::vector<float>& data) {
    auto mem = eng.allocate_memory(layout{shape, data_types::f16, format::bfyx});
    set_values(mem, to_f16(data));
    return mem;
}

memory::ptr make_i32(engine& eng, const ov::PartialShape& shape, const std::vector<int32_t>& data) {
    auto mem = eng.allocate_memory(layout{shape, data_types::i32, format::bfyx});
    set_values(mem, data);
    return mem;
}

/// tests::random_generator only accepts integer bounds, so values are drawn from
/// [-1, 1] with 1/16 resolution and rescaled/shifted here.
std::vector<float> random_vec(tests::random_generator& rg, size_t count, float scale, float bias = 0.0f) {
    auto v = rg.generate_random_1d<float>(count, -1, 1, 16);
    std::vector<float> out(v.begin(), v.end());
    for (auto& x : out) {
        x = x * scale + bias;
    }
    return out;
}

/// Quantizes to f16 and back so that the CPU reference starts from the same values the
/// GPU sees; this keeps the top-k comparison from drifting for near-tied scores.
std::vector<float> round_f16(const std::vector<float>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst[i] = static_cast<float>(ov::float16(src[i]));
    }
    return dst;
}

void rms_norm(std::vector<float>& v, const std::vector<float>& gain, float eps) {
    double sumsq = 0.0;
    for (float x : v) {
        sumsq += static_cast<double>(x) * x;
    }
    const float inv = 1.0f / std::sqrt(static_cast<float>(sumsq / v.size()) + eps);
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = v[i] * inv * gain[i];
    }
}

/// Transcription of transformers' `rotate_half`, kept deliberately literal.
std::vector<float> rotate_half(const std::vector<float>& x) {
    const size_t half = x.size() / 2;
    std::vector<float> out(x.size());
    for (size_t j = 0; j < half; j++) {
        out[j] = -x[half + j];
        out[half + j] = x[j];
    }
    return out;
}

/// Transcription of transformers' `apply_rotary_pos_emb` for a single vector:
///
///     x_rope, x_nope = x[..., :rd], x[..., rd:]
///     x_rope = x_rope * cos + rotate_half(x_rope) * sin
///     out    = cat([x_rope, x_nope])
///
/// `cos_full` / `sin_full` are the [max_positions, rotary] tables in the exact shape
/// the model produces them (`emb = cat((freqs, freqs))`, so the upper half duplicates
/// the lower half). This reference is written against the model formula only - it does
/// not share the packed (cos, sin) pair layout the kernel consumes, so a regression in
/// the kernel's channel pairing cannot hide behind a matching reference.
void apply_rope(std::vector<float>& v,
                const std::vector<float>& cos_full,
                const std::vector<float>& sin_full,
                size_t rotary,
                size_t position) {
    const std::vector<float> x_rope(v.begin(), v.begin() + rotary);
    const auto rotated = rotate_half(x_rope);
    for (size_t j = 0; j < rotary; j++) {
        v[j] = x_rope[j] * cos_full[position * rotary + j] + rotated[j] * sin_full[position * rotary + j];
    }
}

/// Full CPU reference of the QSA pipeline for a single sequence.
/// `history_*` describe tokens already present in the caches (past_len of them),
/// `cur_*` describe the tokens submitted in this call.
struct qsa_reference {
    qsa_dims d;
    size_t past_len = 0;
    size_t q_len = 0;

    std::vector<float> mixed_all;  // [total_len, D] mixed hidden state of the whole sequence
    std::vector<float> key_all;    // [total_len, Hkv*Dh]
    std::vector<float> value_all;  // [total_len, Hkv*Dv]
    std::vector<float> query_cur;  // [q_len, Hq*Dh] (or 2*Hq*Dh when gated)
    std::vector<float> idx_w;      // [(Hidx+1)*Di, D]
    std::vector<float> q_norm;     // [Di]
    std::vector<float> k_norm;     // [Di]
    std::vector<float> rope_cos;   // [max_positions, rotary], `cat(freqs, freqs)` as in the model
    std::vector<float> rope_sin;   // [max_positions, rotary]
    std::vector<float> rotary;     // kernel input: [max_positions, rotary] packed (cos_j, sin_j) pairs
    bool has_gate = false;

    /// Block summaries depend only on the cached hidden states, so they are computed once.
    mutable std::map<size_t, std::vector<float>> summary_memo;

    size_t total_len() const {
        return past_len + q_len;
    }

    /// Indexer projection of one token: returns [Hidx*Di + Di] values.
    std::vector<float> project(size_t abs_pos) const {
        const size_t J = (d.Hidx + 1) * d.Di;
        std::vector<float> out(J, 0.0f);
        for (size_t j = 0; j < J; j++) {
            double acc = 0.0;
            for (size_t k = 0; k < d.D; k++) {
                acc += static_cast<double>(idx_w[j * d.D + k]) * mixed_all[abs_pos * d.D + k];
            }
            out[j] = static_cast<float>(acc);
        }
        return out;
    }

    std::vector<float> summary(size_t block_id) const {
        const auto it = summary_memo.find(block_id);
        if (it != summary_memo.end()) {
            return it->second;
        }
        std::vector<float> pooled(d.Di, 0.0f);
        for (size_t s = 0; s < d.r; s++) {
            const auto proj = project(block_id * d.r + s);
            for (size_t k = 0; k < d.Di; k++) {
                pooled[k] += proj[d.Hidx * d.Di + k];
            }
        }
        for (float& x : pooled) {
            x /= static_cast<float>(d.r);
        }
        rms_norm(pooled, k_norm, d.eps);
        apply_rope(pooled, rope_cos, rope_sin, d.rotary, block_id * d.r);  // block START position
        summary_memo.emplace(block_id, pooled);
        return pooled;
    }

    /// All complete blocks visible from `abs_pos`, sorted the way the kernel must sort
    /// them: score descending, qsa_block_id ascending on ties.
    std::vector<std::pair<float, size_t>> scored_blocks(size_t abs_pos) const {
        const size_t n_complete = (abs_pos + 1) / d.r;
        const auto proj = project(abs_pos);

        std::vector<std::vector<float>> heads(d.Hidx);
        for (size_t h = 0; h < d.Hidx; h++) {
            heads[h].assign(proj.begin() + h * d.Di, proj.begin() + (h + 1) * d.Di);
            rms_norm(heads[h], q_norm, d.eps);
            apply_rope(heads[h], rope_cos, rope_sin, d.rotary, abs_pos);
        }

        std::vector<std::pair<float, size_t>> scored;
        scored.reserve(n_complete);
        for (size_t b = 0; b < n_complete; b++) {
            const auto k = summary(b);
            float total = 0.0f;
            for (size_t h = 0; h < d.Hidx; h++) {
                double dot = 0.0;
                for (size_t j = 0; j < d.Di; j++) {
                    dot += static_cast<double>(heads[h][j]) * k[j];
                }
                total += std::max(0.0f, static_cast<float>(dot));  // ReLU before reduction
            }
            scored.emplace_back(total / std::sqrt(static_cast<float>(d.Di)), b);  // scale after the sum
        }

        std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) {
                return a.first > b.first;  // score descending
            }
            return a.second < b.second;  // qsa_block_id ascending
        });
        return scored;
    }

    std::vector<size_t> select_blocks(size_t abs_pos) const {
        const auto scored = scored_blocks(abs_pos);
        const size_t k = std::min(d.block_topk, scored.size());
        std::vector<size_t> selected;
        selected.reserve(k);
        for (size_t i = 0; i < k; i++) {
            selected.push_back(scored[i].second);
        }
        return selected;
    }

    /// Gap between the last selected and the first rejected block score, relative to the
    /// score magnitude. The block score is `sum_h relu(q_h . k_b)/sqrt(Di)`, so with
    /// symmetric random data a sizeable fraction of the blocks is zeroed by the ReLU on
    /// every head and scores exactly 0. Those blocks are all tied, and when the top-k
    /// boundary falls inside the tied group the FP64 reference and the FP32 systolic
    /// accumulation resolve the tie differently and pick different - equally valid -
    /// blocks. Tokens like that carry no information about the kernel and are skipped.
    /// Infinity means nothing is rejected, i.e. there is no boundary at all.
    float margin_of(const std::vector<std::pair<float, size_t>>& scored) const {
        if (scored.size() <= d.block_topk) {
            return std::numeric_limits<float>::infinity();
        }
        const float lo = scored[d.block_topk - 1].first;
        const float hi = scored[d.block_topk].first;
        return (lo - hi) / std::max(std::fabs(lo), 1e-3f);
    }

    float selection_margin(size_t abs_pos) const {
        return margin_of(scored_blocks(abs_pos));
    }

    /// `margins_out`, when given, receives the per-token top-k margin. Scoring a token is
    /// by far the most expensive part of this reference (O(n_complete * Hidx * Di)), so
    /// producing it here avoids a second full pass for sequences of a few thousand tokens.
    std::vector<float> attention(std::vector<float>* margins_out = nullptr) const {
        const size_t q_stride = has_gate ? 2 * d.Hq * d.Dh : d.Hq * d.Dh;
        std::vector<float> out(q_len * d.Hq * d.Dv, 0.0f);
        if (margins_out != nullptr) {
            margins_out->assign(q_len, 0.0f);
        }

        for (size_t t = 0; t < q_len; t++) {
            const size_t abs_pos = past_len + t;

            // The selection is per query token and shared by every query head.
            const auto scored = scored_blocks(abs_pos);
            if (margins_out != nullptr) {
                (*margins_out)[t] = margin_of(scored);
            }
            std::vector<size_t> positions;
            for (size_t i = 0; i < std::min(d.block_topk, scored.size()); i++) {
                for (size_t s = 0; s < d.r; s++) {
                    positions.push_back(scored[i].second * d.r + s);
                }
            }
            for (size_t p = ((abs_pos + 1) / d.r) * d.r; p <= abs_pos; p++) {
                positions.push_back(p);  // always-visible tail of the incomplete block
            }

            for (size_t h = 0; h < d.Hq; h++) {
                const size_t kv_head = h / (d.Hq / d.Hkv);
                const size_t q_base = t * q_stride + h * (has_gate ? 2 * d.Dh : d.Dh);

                std::vector<float> logits;
                logits.reserve(positions.size());
                for (size_t p : positions) {
                    double dot = 0.0;
                    for (size_t j = 0; j < d.Dh; j++) {
                        dot += static_cast<double>(query_cur[q_base + j]) * key_all[p * d.Hkv * d.Dh + kv_head * d.Dh + j];
                    }
                    logits.push_back(static_cast<float>(dot) / std::sqrt(static_cast<float>(d.Dh)));
                }

                const float m = *std::max_element(logits.begin(), logits.end());
                float sum = 0.0f;
                for (float& x : logits) {
                    x = std::exp(x - m);
                    sum += x;
                }
                for (size_t j = 0; j < d.Dv; j++) {
                    double acc = 0.0;
                    for (size_t i = 0; i < positions.size(); i++) {
                        acc += static_cast<double>(logits[i]) * value_all[positions[i] * d.Hkv * d.Dv + kv_head * d.Dv + j];
                    }
                    float v = static_cast<float>(acc / sum);
                    if (has_gate) {
                        const float g = query_cur[q_base + d.Dh + j];
                        v *= 1.0f / (1.0f + std::exp(-g));
                    }
                    out[(t * d.Hq + h) * d.Dv + j] = v;
                }
            }
        }
        return out;
    }
};

/// Builds the topology, runs QSA on the GPU and returns the output.
std::vector<float> run_qsa(const qsa_reference& ref, bool has_gate) {
    auto& engine = get_test_engine();
    const auto& d = ref.d;
    const auto config_attrs = make_config(d, has_gate);

    const size_t total = ref.total_len();
    const size_t num_pages = (total + d.pa_block - 1) / d.pa_block;
    const size_t summaries_per_page = d.pa_block / d.r;

    // Slice out the tokens submitted in this call.
    const auto slice = [&](const std::vector<float>& src, size_t stride) {
        return std::vector<float>(src.begin() + ref.past_len * stride, src.begin() + total * stride);
    };
    const auto cur_mixed = slice(ref.mixed_all, d.D);
    const auto cur_key = slice(ref.key_all, d.Hkv * d.Dh);
    const auto cur_value = slice(ref.value_all, d.Hkv * d.Dv);

    // Caches: history tokens are pre-populated exactly the way Q0 / Q1 would have.
    std::vector<float> key_cache(num_pages * d.Hkv * d.pa_block * d.Dh, 0.0f);
    std::vector<float> value_cache(num_pages * d.Hkv * d.pa_block * d.Dv, 0.0f);
    std::vector<float> raw_k_cache(num_pages * d.pa_block * d.Di, 0.0f);
    std::vector<float> summary_cache(num_pages * summaries_per_page * d.Di, 0.0f);

    for (size_t p = 0; p < ref.past_len; p++) {
        const size_t page = p / d.pa_block;
        const size_t slot = p % d.pa_block;
        for (size_t h = 0; h < d.Hkv; h++) {
            for (size_t j = 0; j < d.Dh; j++) {
                key_cache[((page * d.Hkv + h) * d.pa_block + slot) * d.Dh + j] = ref.key_all[p * d.Hkv * d.Dh + h * d.Dh + j];
            }
            for (size_t j = 0; j < d.Dv; j++) {
                value_cache[((page * d.Hkv + h) * d.pa_block + slot) * d.Dv + j] = ref.value_all[p * d.Hkv * d.Dv + h * d.Dv + j];
            }
        }
        const auto proj = ref.project(p);
        for (size_t j = 0; j < d.Di; j++) {
            raw_k_cache[(page * d.pa_block + slot) * d.Di + j] = proj[d.Hidx * d.Di + j];
        }
    }
    for (size_t b = 0; b * d.r + d.r <= ref.past_len; b++) {
        const auto s = ref.summary(b);
        const size_t page = b / summaries_per_page;
        const size_t slot = b % summaries_per_page;
        for (size_t j = 0; j < d.Di; j++) {
            summary_cache[(page * summaries_per_page + slot) * d.Di + j] = s[j];
        }
    }

    std::vector<int32_t> block_indices(num_pages);
    std::iota(block_indices.begin(), block_indices.end(), 0);
    const std::vector<int32_t> block_indices_begins = {0, static_cast<int32_t>(num_pages)};
    const std::vector<int32_t> past_lens = {static_cast<int32_t>(ref.past_len)};
    const std::vector<int32_t> subsequence_begins = {0, static_cast<int32_t>(ref.q_len)};
    std::vector<int32_t> position_ids(ref.q_len);
    for (size_t t = 0; t < ref.q_len; t++) {
        position_ids[t] = static_cast<int32_t>(ref.past_len + t);
    }

    const auto q_width = has_gate ? 2 * d.Hq * d.Dh : d.Hq * d.Dh;
    const size_t J = (d.Hidx + 1) * d.Di;

    topology topology;
    topology.add(input_layout("mixed", layout{{static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.D)}, data_types::f16, format::bfyx}));
    topology.add(input_layout("query", layout{{static_cast<int64_t>(ref.q_len), static_cast<int64_t>(q_width)}, data_types::f16, format::bfyx}));
    topology.add(input_layout("key", layout{{static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.Hkv * d.Dh)}, data_types::f16, format::bfyx}));
    topology.add(input_layout("value", layout{{static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.Hkv * d.Dv)}, data_types::f16, format::bfyx}));
    topology.add(data("past_lens", make_i32(engine, {1}, past_lens)));
    topology.add(data("subsequence_begins", make_i32(engine, {2}, subsequence_begins)));
    topology.add(data("block_indices", make_i32(engine, {static_cast<int64_t>(num_pages)}, block_indices)));
    topology.add(data("block_indices_begins", make_i32(engine, {2}, block_indices_begins)));
    topology.add(input_layout("key_cache",
                              layout{{static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block),
                                      static_cast<int64_t>(d.Dh)},
                                     data_types::f16,
                                     format::bfyx}));
    topology.add(input_layout("value_cache",
                              layout{{static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block),
                                      static_cast<int64_t>(d.Dv)},
                                     data_types::f16,
                                     format::bfyx}));
    topology.add(input_layout("raw_k_cache",
                              layout{{static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Di)},
                                     data_types::f16,
                                     format::bfyx}));
    topology.add(input_layout("summary_cache",
                              layout{{static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)},
                                     data_types::f16,
                                     format::bfyx}));
    topology.add(data("idx_w", make_f16(engine, {static_cast<int64_t>(J), static_cast<int64_t>(d.D)}, ref.idx_w)));
    topology.add(data("q_norm", make_f16(engine, {static_cast<int64_t>(d.Di)}, ref.q_norm)));
    topology.add(data("k_norm", make_f16(engine, {static_cast<int64_t>(d.Di)}, ref.k_norm)));
    topology.add(data("rotary", make_f16(engine, {static_cast<int64_t>(d.max_positions), static_cast<int64_t>(d.rotary)}, ref.rotary)));
    topology.add(data("position_ids", make_i32(engine, {static_cast<int64_t>(ref.q_len)}, position_ids)));

    const std::vector<input_info> inputs = {input_info("mixed"),
                                            input_info("query"),
                                            input_info("key"),
                                            input_info("value"),
                                            input_info("past_lens"),
                                            input_info("subsequence_begins"),
                                            input_info("block_indices"),
                                            input_info("block_indices_begins"),
                                            input_info("key_cache"),
                                            input_info("value_cache"),
                                            input_info("raw_k_cache"),
                                            input_info("summary_cache"),
                                            input_info("idx_w"),
                                            input_info("q_norm"),
                                            input_info("k_norm"),
                                            input_info("rotary"),
                                            input_info("position_ids")};
    static_assert(QSAIn::COUNT == 17, "keep the test input order in sync with QSA::InputIdx");

    topology.add(qsa("qsa", inputs, config_attrs));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::optimize_data(true));
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, cfg);

    network.set_input_data("mixed", make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.D)}, cur_mixed));
    network.set_input_data("query", make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(q_width)}, ref.query_cur));
    network.set_input_data("key", make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.Hkv * d.Dh)}, cur_key));
    network.set_input_data("value", make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.Hkv * d.Dv)}, cur_value));
    network.set_input_data("key_cache",
                           make_f16(engine,
                                    {static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block),
                                     static_cast<int64_t>(d.Dh)},
                                    key_cache));
    network.set_input_data("value_cache",
                           make_f16(engine,
                                    {static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block),
                                     static_cast<int64_t>(d.Dv)},
                                    value_cache));
    network.set_input_data("raw_k_cache",
                           make_f16(engine,
                                    {static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Di)},
                                    raw_k_cache));
    network.set_input_data("summary_cache",
                           make_f16(engine,
                                    {static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)},
                                    summary_cache));

    auto outputs = network.execute();
    cldnn::mem_lock<ov::float16> out(outputs.at("qsa").get_memory(), get_test_stream());
    std::vector<float> result(out.size());
    for (size_t i = 0; i < out.size(); i++) {
        result[i] = static_cast<float>(out[i]);
    }
    return result;
}

/// Runs QSA once over a BATCH of independent sequences in a single network.execute()
/// call - the continuous-batching / QSAStage::MIXED scenario when the batch mixes a
/// q_len > 1 (prefill or chunked-prefill-continuation) entry with a q_len == 1 (decode)
/// entry. Every `ref` in `refs` must share one QSA layer's weights (see
/// `make_reference_sharing_weights`); sequences are otherwise fully independent (QSA has
/// no cross-sequence interaction) and each gets its own contiguous range of PA pages in a
/// cache shared by the whole batch, exactly as a real paged-attention allocator would lay
/// out unrelated requests. Returns the flat [total_tokens, Hq*Dv] output, tokens ordered
/// the same way `subsequence_begins` orders them (sequence 0's tokens, then sequence 1's,
/// ...).
std::vector<float> run_qsa_batch(const std::vector<qsa_reference>& refs, bool has_gate) {
    auto& engine = get_test_engine();
    if (refs.empty()) {
        ADD_FAILURE() << "run_qsa_batch requires at least one sequence";
        return {};
    }
    const auto& d = refs.front().d;
    const size_t num_seqs = refs.size();
    const size_t summaries_per_page = d.pa_block / d.r;

    size_t total_tokens = 0;
    size_t total_pages = 0;
    std::vector<size_t> page_begin(num_seqs);
    std::vector<size_t> num_pages_of(num_seqs);
    for (size_t s = 0; s < num_seqs; s++) {
        total_tokens += refs[s].q_len;
        page_begin[s] = total_pages;
        num_pages_of[s] = (refs[s].total_len() + d.pa_block - 1) / d.pa_block;
        total_pages += num_pages_of[s];
    }

    std::vector<float> mixed_cur, query_cur, key_cur, value_cur;
    std::vector<float> key_cache(total_pages * d.Hkv * d.pa_block * d.Dh, 0.0f);
    std::vector<float> value_cache(total_pages * d.Hkv * d.pa_block * d.Dv, 0.0f);
    std::vector<float> raw_k_cache(total_pages * d.pa_block * d.Di, 0.0f);
    std::vector<float> summary_cache(total_pages * summaries_per_page * d.Di, 0.0f);
    std::vector<int32_t> past_lens(num_seqs);
    std::vector<int32_t> subsequence_begins(num_seqs + 1, 0);
    std::vector<int32_t> block_indices_begins(num_seqs + 1, 0);
    std::vector<int32_t> block_indices(total_pages);
    std::iota(block_indices.begin(), block_indices.end(), 0);
    std::vector<int32_t> position_ids;
    position_ids.reserve(total_tokens);

    for (size_t s = 0; s < num_seqs; s++) {
        const auto& ref = refs[s];
        if (ref.d.D != d.D || ref.d.Hq != d.Hq || ref.d.Hkv != d.Hkv) {
            ADD_FAILURE() << "run_qsa_batch requires every sequence to share one QSA config";
            return {};
        }
        const size_t total = ref.total_len();
        past_lens[s] = static_cast<int32_t>(ref.past_len);
        subsequence_begins[s + 1] = subsequence_begins[s] + static_cast<int32_t>(ref.q_len);
        block_indices_begins[s + 1] = block_indices_begins[s] + static_cast<int32_t>(num_pages_of[s]);

        const auto slice = [&](const std::vector<float>& src, size_t stride) {
            return std::vector<float>(src.begin() + ref.past_len * stride, src.begin() + total * stride);
        };
        const auto cm = slice(ref.mixed_all, d.D);
        const auto ck = slice(ref.key_all, d.Hkv * d.Dh);
        const auto cv = slice(ref.value_all, d.Hkv * d.Dv);
        mixed_cur.insert(mixed_cur.end(), cm.begin(), cm.end());
        key_cur.insert(key_cur.end(), ck.begin(), ck.end());
        value_cur.insert(value_cur.end(), cv.begin(), cv.end());
        query_cur.insert(query_cur.end(), ref.query_cur.begin(), ref.query_cur.end());

        for (size_t t = 0; t < ref.q_len; t++) {
            position_ids.push_back(static_cast<int32_t>(ref.past_len + t));
        }

        // History tokens are pre-populated exactly the way Q0 / Q1 would have, at this
        // sequence's own page range so unrelated sequences never alias.
        for (size_t p = 0; p < ref.past_len; p++) {
            const size_t page = page_begin[s] + p / d.pa_block;
            const size_t slot = p % d.pa_block;
            for (size_t h = 0; h < d.Hkv; h++) {
                for (size_t j = 0; j < d.Dh; j++) {
                    key_cache[((page * d.Hkv + h) * d.pa_block + slot) * d.Dh + j] = ref.key_all[p * d.Hkv * d.Dh + h * d.Dh + j];
                }
                for (size_t j = 0; j < d.Dv; j++) {
                    value_cache[((page * d.Hkv + h) * d.pa_block + slot) * d.Dv + j] = ref.value_all[p * d.Hkv * d.Dv + h * d.Dv + j];
                }
            }
            const auto proj = ref.project(p);
            for (size_t j = 0; j < d.Di; j++) {
                raw_k_cache[(page * d.pa_block + slot) * d.Di + j] = proj[d.Hidx * d.Di + j];
            }
        }
        for (size_t b = 0; b * d.r + d.r <= ref.past_len; b++) {
            const auto sm = ref.summary(b);
            const size_t page = page_begin[s] + b / summaries_per_page;
            const size_t slot = b % summaries_per_page;
            for (size_t j = 0; j < d.Di; j++) {
                summary_cache[(page * summaries_per_page + slot) * d.Di + j] = sm[j];
            }
        }
    }

    const auto q_width = has_gate ? 2 * d.Hq * d.Dh : d.Hq * d.Dh;
    const size_t J = (d.Hidx + 1) * d.Di;
    const auto& w = refs.front();  // shared layer weights

    topology topology;
    topology.add(input_layout("mixed", layout{{static_cast<int64_t>(total_tokens), static_cast<int64_t>(d.D)}, data_types::f16, format::bfyx}));
    topology.add(input_layout("query", layout{{static_cast<int64_t>(total_tokens), static_cast<int64_t>(q_width)}, data_types::f16, format::bfyx}));
    topology.add(input_layout("key", layout{{static_cast<int64_t>(total_tokens), static_cast<int64_t>(d.Hkv * d.Dh)}, data_types::f16, format::bfyx}));
    topology.add(input_layout("value", layout{{static_cast<int64_t>(total_tokens), static_cast<int64_t>(d.Hkv * d.Dv)}, data_types::f16, format::bfyx}));
    topology.add(data("past_lens", make_i32(engine, {static_cast<int64_t>(num_seqs)}, past_lens)));
    topology.add(data("subsequence_begins", make_i32(engine, {static_cast<int64_t>(num_seqs + 1)}, subsequence_begins)));
    topology.add(data("block_indices", make_i32(engine, {static_cast<int64_t>(total_pages)}, block_indices)));
    topology.add(data("block_indices_begins", make_i32(engine, {static_cast<int64_t>(num_seqs + 1)}, block_indices_begins)));
    topology.add(input_layout("key_cache",
                              layout{{static_cast<int64_t>(total_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block),
                                      static_cast<int64_t>(d.Dh)},
                                     data_types::f16,
                                     format::bfyx}));
    topology.add(input_layout("value_cache",
                              layout{{static_cast<int64_t>(total_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block),
                                      static_cast<int64_t>(d.Dv)},
                                     data_types::f16,
                                     format::bfyx}));
    topology.add(input_layout("raw_k_cache",
                              layout{{static_cast<int64_t>(total_pages), 1, static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Di)},
                                     data_types::f16,
                                     format::bfyx}));
    topology.add(input_layout("summary_cache",
                              layout{{static_cast<int64_t>(total_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)},
                                     data_types::f16,
                                     format::bfyx}));
    topology.add(data("idx_w", make_f16(engine, {static_cast<int64_t>(J), static_cast<int64_t>(d.D)}, w.idx_w)));
    topology.add(data("q_norm", make_f16(engine, {static_cast<int64_t>(d.Di)}, w.q_norm)));
    topology.add(data("k_norm", make_f16(engine, {static_cast<int64_t>(d.Di)}, w.k_norm)));
    topology.add(data("rotary", make_f16(engine, {static_cast<int64_t>(d.max_positions), static_cast<int64_t>(d.rotary)}, w.rotary)));
    topology.add(data("position_ids", make_i32(engine, {static_cast<int64_t>(total_tokens)}, position_ids)));

    const std::vector<input_info> inputs = {input_info("mixed"),
                                            input_info("query"),
                                            input_info("key"),
                                            input_info("value"),
                                            input_info("past_lens"),
                                            input_info("subsequence_begins"),
                                            input_info("block_indices"),
                                            input_info("block_indices_begins"),
                                            input_info("key_cache"),
                                            input_info("value_cache"),
                                            input_info("raw_k_cache"),
                                            input_info("summary_cache"),
                                            input_info("idx_w"),
                                            input_info("q_norm"),
                                            input_info("k_norm"),
                                            input_info("rotary"),
                                            input_info("position_ids")};
    static_assert(QSAIn::COUNT == 17, "keep the test input order in sync with QSA::InputIdx");

    const auto config_attrs = make_config(d, has_gate);
    topology.add(qsa("qsa", inputs, config_attrs));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::optimize_data(true));
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, cfg);

    network.set_input_data("mixed", make_f16(engine, {static_cast<int64_t>(total_tokens), static_cast<int64_t>(d.D)}, mixed_cur));
    network.set_input_data("query", make_f16(engine, {static_cast<int64_t>(total_tokens), static_cast<int64_t>(q_width)}, query_cur));
    network.set_input_data("key", make_f16(engine, {static_cast<int64_t>(total_tokens), static_cast<int64_t>(d.Hkv * d.Dh)}, key_cur));
    network.set_input_data("value", make_f16(engine, {static_cast<int64_t>(total_tokens), static_cast<int64_t>(d.Hkv * d.Dv)}, value_cur));
    network.set_input_data("key_cache",
                           make_f16(engine,
                                    {static_cast<int64_t>(total_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block),
                                     static_cast<int64_t>(d.Dh)},
                                    key_cache));
    network.set_input_data("value_cache",
                           make_f16(engine,
                                    {static_cast<int64_t>(total_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block),
                                     static_cast<int64_t>(d.Dv)},
                                    value_cache));
    network.set_input_data("raw_k_cache",
                           make_f16(engine,
                                    {static_cast<int64_t>(total_pages), 1, static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Di)},
                                    raw_k_cache));
    network.set_input_data("summary_cache",
                           make_f16(engine,
                                    {static_cast<int64_t>(total_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)},
                                    summary_cache));

    auto outputs = network.execute();
    cldnn::mem_lock<ov::float16> out(outputs.at("qsa").get_memory(), get_test_stream());
    std::vector<float> result(out.size());
    for (size_t i = 0; i < out.size(); i++) {
        result[i] = static_cast<float>(out[i]);
    }
    return result;
}

qsa_reference make_reference_once(tests::random_generator& rg, const qsa_dims& d, size_t past_len, size_t q_len, bool has_gate) {
    qsa_reference ref;
    ref.d = d;
    ref.past_len = past_len;
    ref.q_len = q_len;
    ref.has_gate = has_gate;

    const size_t total = past_len + q_len;
    ref.mixed_all = round_f16(random_vec(rg, total * d.D, 1.0f));
    ref.key_all = round_f16(random_vec(rg, total * d.Hkv * d.Dh, 1.0f));
    ref.value_all = round_f16(random_vec(rg, total * d.Hkv * d.Dv, 1.0f));
    ref.query_cur = round_f16(random_vec(rg, q_len * (has_gate ? 2 : 1) * d.Hq * d.Dh, 1.0f));
    ref.idx_w = round_f16(random_vec(rg, (d.Hidx + 1) * d.Di * d.D, 0.2f));
    ref.q_norm = round_f16(random_vec(rg, d.Di, 0.5f, 1.0f));
    ref.k_norm = round_f16(random_vec(rg, d.Di, 0.5f, 1.0f));

    // Build the cos/sin tables the way the model does: `freqs` has rotary/2 entries and
    // `emb = cat((freqs, freqs))`, so cos[j] == cos[j + rotary/2]. mRoPE section handling
    // is a host-side concern and only changes the table contents, not this layout.
    const size_t half = d.rotary / 2;
    ref.rope_cos.resize(d.max_positions * d.rotary);
    ref.rope_sin.resize(d.max_positions * d.rotary);
    for (size_t p = 0; p < d.max_positions; p++) {
        for (size_t j = 0; j < half; j++) {
            const float theta = static_cast<float>(p) / std::pow(10000.0f, 2.0f * j / static_cast<float>(d.rotary));
            const float c = static_cast<float>(ov::float16(std::cos(theta)));
            const float s = static_cast<float>(ov::float16(std::sin(theta)));
            ref.rope_cos[p * d.rotary + j] = c;
            ref.rope_cos[p * d.rotary + half + j] = c;
            ref.rope_sin[p * d.rotary + j] = s;
            ref.rope_sin[p * d.rotary + half + j] = s;
        }
    }

    // The kernel takes the rotary/2 distinct frequencies packed as (cos_j, sin_j) pairs,
    // since the duplicated upper half of the model's tables carries no information.
    ref.rotary.assign(d.max_positions * d.rotary, 0.0f);
    for (size_t p = 0; p < d.max_positions; p++) {
        for (size_t j = 0; j < half; j++) {
            ref.rotary[p * d.rotary + 2 * j] = ref.rope_cos[p * d.rotary + j];
            ref.rotary[p * d.rotary + 2 * j + 1] = ref.rope_sin[p * d.rotary + j];
        }
    }
    return ref;
}

/// A token is only compared when its top-k boundary is separated by at least this much;
/// see `qsa_reference::margin_of`. The floor is set by precision, not by taste: Q1 writes
/// the indexer queries and the block summaries back to the caches as f16, so the scores
/// the GPU ranks carry ~1e-3 of relative quantization noise on top of this FP64 reference.
/// Anything closer than that at the boundary can legitimately rank either way.
constexpr float kMinSelectionMargin = 0.02f;

/// Tokens that must survive the margin filter for a comparison to be meaningful. Long
/// sequences have hundreds; requiring a *fraction* of them would be wrong, because with a
/// thousand candidate blocks the natural gap between the k-th and (k+1)-th score is only a
/// few percent and a large share of tokens legitimately sits near the boundary.
constexpr size_t kMinComparedTokens = 8;

qsa_reference make_reference(tests::random_generator& rg, const qsa_dims& d, size_t past_len, size_t q_len, bool has_gate) {
    if (q_len > 1) {
        return make_reference_once(rg, d, past_len, q_len, has_gate);
    }
    // A single decode step has exactly one token, so there is nothing to filter: redraw
    // until its boundary is unambiguous. Scoring one token is cheap, and `rg` is seeded
    // per suite so the retries stay deterministic.
    qsa_reference ref;
    for (int attempt = 0; attempt < 64; attempt++) {
        ref = make_reference_once(rg, d, past_len, q_len, has_gate);
        if (ref.selection_margin(past_len) >= kMinSelectionMargin) {
            return ref;
        }
    }
    ADD_FAILURE() << "could not draw a decode step whose top-k boundary is separated by " << kMinSelectionMargin;
    return ref;
}

/// Draws a second sequence's runtime data (hidden states, KV, query) while reusing the
/// SAME layer weights (idx_w, RMSNorm gains, rotary table) as `shared`. One real QSA
/// primitive is one layer serving every sequence in a batch with one set of weights, so a
/// multi-sequence test must not let each sequence draw its own - that would silently be
/// testing N independent layers instead of N requests against one layer.
qsa_reference make_reference_sharing_weights(tests::random_generator& rg, const qsa_reference& shared, size_t past_len, size_t q_len, bool has_gate) {
    const auto assign_shared_weights = [&](qsa_reference& ref) {
        ref.idx_w = shared.idx_w;
        ref.q_norm = shared.q_norm;
        ref.k_norm = shared.k_norm;
        ref.rope_cos = shared.rope_cos;
        ref.rope_sin = shared.rope_sin;
        ref.rotary = shared.rotary;
    };
    if (q_len > 1) {
        auto ref = make_reference_once(rg, shared.d, past_len, q_len, has_gate);
        assign_shared_weights(ref);
        return ref;
    }
    qsa_reference ref;
    for (int attempt = 0; attempt < 64; attempt++) {
        ref = make_reference_once(rg, shared.d, past_len, q_len, has_gate);
        assign_shared_weights(ref);
        if (ref.selection_margin(past_len) >= kMinSelectionMargin) {
            return ref;
        }
    }
    ADD_FAILURE() << "could not draw a decode step whose top-k boundary is separated by " << kMinSelectionMargin;
    return ref;
}

void expect_close(const std::vector<float>& got, const std::vector<float>& expected, float tol) {
    ASSERT_EQ(got.size(), expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        ASSERT_NEAR(got[i], expected[i], tol) << "mismatch at " << i;
    }
}

/// Runs QSA and compares against the reference token by token, skipping the tokens whose
/// top-k boundary is a tie (those measure FP association order, not the kernel). Fails if
/// too few tokens survive that filter, so a degenerate configuration cannot pass silently.
void expect_qsa_matches(const qsa_reference& ref, bool has_gate, float tol) {
    const auto got = run_qsa(ref, has_gate);
    std::vector<float> margins;
    const auto expected = ref.attention(&margins);
    ASSERT_EQ(got.size(), expected.size());

    const size_t width = ref.d.Hq * ref.d.Dv;
    size_t compared = 0;
    for (size_t t = 0; t < ref.q_len; t++) {
        if (margins[t] < kMinSelectionMargin) {
            continue;
        }
        compared++;
        for (size_t i = t * width; i < (t + 1) * width; i++) {
            ASSERT_NEAR(got[i], expected[i], tol) << "mismatch at " << i << " (token " << t << ")";
        }
    }
    ASSERT_GE(compared, std::min(ref.q_len, kMinComparedTokens))
        << "only " << compared << " of " << ref.q_len
        << " tokens had an unambiguous top-k boundary; the configuration is too degenerate for this comparison to mean anything";
}

/// Overrides the QSA Q2 BLOCK_SCORES scratch budget (see qsa.cpp's `score_budget_bytes()`)
/// for the lifetime of the guard, then restores the production default. Production code
/// never sets this env var; it exists purely so a regression test can force the
/// query-row-chunked path (QSAScoreChunk) to actually run more than once per call without
/// allocating a multi-GB buffer.
class score_budget_override {
public:
    explicit score_budget_override(size_t bytes) {
        const auto value = std::to_string(bytes);
#ifdef _WIN32
        _putenv_s("QSA_TEST_SCORE_BUDGET_BYTES", value.c_str());
#else
        ::setenv("QSA_TEST_SCORE_BUDGET_BYTES", value.c_str(), 1);
#endif
    }
    ~score_budget_override() {
#ifdef _WIN32
        _putenv_s("QSA_TEST_SCORE_BUDGET_BYTES", "");
#else
        ::unsetenv("QSA_TEST_SCORE_BUDGET_BYTES");
#endif
    }
    score_budget_override(const score_budget_override&) = delete;
    score_budget_override& operator=(const score_budget_override&) = delete;
};

TEST(qsa_gpu_test, prefill_single_sequence) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 0, 40);
    auto ref = make_reference(rg, d, /*past_len=*/0, /*q_len=*/40, /*has_gate=*/false);
    expect_qsa_matches(ref, false, 5e-2);
}

TEST(qsa_gpu_test, prefill_shorter_than_one_block) {
    // q_len < compress_ratio: no complete block exists, every visible token must come
    // from the tail path and top-k must be empty rather than padded with illegal blocks.
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    auto ref = make_reference(rg, d, 0, 3, false);
    expect_qsa_matches(ref, false, 5e-2);
}

TEST(qsa_gpu_test, decode_appends_single_token) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 41, 1);
    auto ref = make_reference(rg, d, /*past_len=*/41, /*q_len=*/1, /*has_gate=*/false);
    expect_qsa_matches(ref, false, 5e-2);
}

TEST(qsa_gpu_test, decode_completes_a_block) {
    // The appended token fills the last slot of a QSA block, so Q1 must pool the block
    // from raw-K (partly cached, partly just produced) and emit its summary.
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 43, 1);
    auto ref = make_reference(rg, d, /*past_len=*/43, /*q_len=*/1, /*has_gate=*/false);
    ASSERT_EQ((ref.past_len + ref.q_len) % d.r, 0u);
    expect_close(run_qsa(ref, false), ref.attention(), 5e-2f);
}

TEST(qsa_gpu_test, output_gate_from_fused_q_proj) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 0, 40);
    auto ref = make_reference(rg, d, 0, 40, /*has_gate=*/true);
    expect_qsa_matches(ref, true, 5e-2);
}

TEST(qsa_gpu_test, prefill_and_decode_agree) {
    // The prefill fused Q2 variant and the decode partition + finalization variant must
    // select the same blocks; comparing the last prefill token against a decode step
    // that ends at the same absolute position exercises exactly that.
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 0, 40);
    auto prefill = make_reference(rg, d, 0, 40, false);
    const auto prefill_out = run_qsa(prefill, false);

    qsa_reference decode = prefill;
    decode.past_len = 39;
    decode.q_len = 1;
    decode.query_cur.assign(prefill.query_cur.end() - static_cast<long>(d.Hq * d.Dh), prefill.query_cur.end());
    const auto decode_out = run_qsa(decode, false);

    const size_t offset = 39 * d.Hq * d.Dv;
    ASSERT_EQ(decode_out.size(), d.Hq * d.Dv);
    for (size_t i = 0; i < decode_out.size(); i++) {
        ASSERT_NEAR(decode_out[i], prefill_out[offset + i], 5e-2f) << "prefill/decode divergence at " << i;
    }
}

// --- systolic (cm_dpas) prefill scoring ------------------------------------

TEST(qsa_gpu_test, prefill_dpas_scoring_matches_reference) {
    // Exercises qsa_score_topk_dpas: the (block x query) score matrix comes out of the
    // systolic array instead of lane-reduction dot products. The ReLU is still applied
    // per indexer head before the head reduction, and 1/sqrt(Di) after it, so the result
    // must match the same FP32 reference as the scalar path.
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const auto d = dpas_dims();
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 0, 64);
    auto ref = make_reference(rg, d, /*past_len=*/0, /*q_len=*/64, /*has_gate=*/false);
    expect_qsa_matches(ref, false, 5e-2);
}

TEST(qsa_gpu_test, prefill_dpas_scoring_partial_query_tile) {
    // q_len is not a multiple of the 16-wide dpas query tile: the trailing tile is
    // partially valid and the descriptor's surface height must zero the missing rows.
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const auto d = dpas_dims();
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 0, 53);
    auto ref = make_reference(rg, d, 0, 53, false);
    expect_qsa_matches(ref, false, 5e-2);
}

TEST(qsa_gpu_test, dpas_chunked_prefill_with_past) {
    // Chunked prefill: past_len > 0 so the query tile's blocks resolve through the page
    // table at a non-zero offset, and the queries of one tile no longer share a selection
    // (their n_complete differs), which is exactly what the union + per-query mask covers.
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const auto d = dpas_dims();
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 36, 40);
    auto ref = make_reference(rg, d, /*past_len=*/36, /*q_len=*/40, /*has_gate=*/true);
    expect_qsa_matches(ref, true, 5e-2);
}

TEST(qsa_gpu_test, dpas_score_multi_chunk_matches_single_chunk) {
    // Forces qsa.cpp's query-row-chunked BLOCK_SCORES path (QSAScoreChunk / score_chunks)
    // to actually run more than once per call. Without shrinking the budget, no test in
    // this file gets anywhere near the production 128 MiB default, so the chunk loop in
    // execute() always plans exactly one chunk and a chunk-boundary regression (stale
    // dispatch data reused across chunks, wrong re-anchored tile map, ...) would stay
    // invisible until someone tried a 100K+ token prefill by hand.
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const auto d = dpas_dims();
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 0, 64);
    auto ref = make_reference(rg, d, /*past_len=*/0, /*q_len=*/64, /*has_gate=*/false);

    // One f32 BLOCK_SCORES row is max_blocks * 4 bytes; a 64-byte budget is well under one
    // row, so every chunk covers exactly 1 query row and a 64-token prefill plans 64
    // chunks instead of the usual 1.
    score_budget_override guard(64);
    expect_qsa_matches(ref, false, 5e-2);
}

TEST(qsa_gpu_test, dpas_prefill_and_scalar_decode_agree) {
    // The systolic prefill selector and the scalar decode partition + finalization
    // selector must pick the same blocks; comparing the last prefill token against a
    // decode step ending at the same absolute position checks exactly that.
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const auto d = dpas_dims();
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 0, 64);
    auto prefill = make_reference(rg, d, 0, 64, false);
    const auto prefill_out = run_qsa(prefill, false);

    qsa_reference decode = prefill;
    decode.past_len = 63;
    decode.q_len = 1;
    decode.query_cur.assign(prefill.query_cur.end() - static_cast<long>(d.Hq * d.Dh), prefill.query_cur.end());
    const auto decode_out = run_qsa(decode, false);

    const size_t offset = 63 * d.Hq * d.Dv;
    ASSERT_EQ(decode_out.size(), d.Hq * d.Dv);
    for (size_t i = 0; i < decode_out.size(); i++) {
        ASSERT_NEAR(decode_out[i], prefill_out[offset + i], 5e-2f) << "dpas/scalar selection divergence at " << i;
    }
}

// --- continuous batching (QSAStage::MIXED) ---------------------------------

TEST(qsa_gpu_test, mixed_batch_prefill_and_decode_together) {
    // Continuous batching: one network.execute() call carries a multi-token
    // chunked-prefill-continuation sequence AND a single-token decode sequence at the
    // same time (any_multi && any_single in qsa.cpp::update_rt_params -> QSAStage::MIXED),
    // sharing one physical page table exactly as a real scheduler packing unrelated
    // requests would. Both sequences use the SAME layer weights (idx_w, norm gains,
    // rotary table) but independent hidden states/KV/query and independent PA page
    // ranges, so a bug that let one sequence's dispatch read another's data (wrong
    // subsequence_begins/block_indices_begins indexing in Q0/Q1/Q2/Q3) would show up as a
    // numeric mismatch here even though every existing test in this file only ever
    // submits num_seqs == 1.
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const auto d = prod_gqa_dims(200);
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 40, 37);
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 63, 1);

    // seq0: chunked-prefill continuation (past_len > 0, q_len > 1) -> any_multi.
    auto seq0 = make_reference(rg, d, /*past_len=*/40, /*q_len=*/37, /*has_gate=*/true);
    // seq1: decode append (q_len == 1) -> any_single. Shares seq0's weights: this is one
    // QSA layer serving two requests, not two independently-configured layers.
    auto seq1 = make_reference_sharing_weights(rg, seq0, /*past_len=*/63, /*q_len=*/1, /*has_gate=*/true);

    const auto got = run_qsa_batch({seq0, seq1}, /*has_gate=*/true);

    const size_t width = d.Hq * d.Dv;
    ASSERT_EQ(got.size(), (seq0.q_len + seq1.q_len) * width);

    std::vector<float> margins0, margins1;
    const auto expected0 = seq0.attention(&margins0);
    const auto expected1 = seq1.attention(&margins1);

    size_t compared = 0;
    for (size_t t = 0; t < seq0.q_len; t++) {
        if (margins0[t] < kMinSelectionMargin) {
            continue;
        }
        compared++;
        for (size_t i = t * width; i < (t + 1) * width; i++) {
            ASSERT_NEAR(got[i], expected0[i], 8e-2f) << "seq0 (prefill) mismatch at token " << t << " elem " << i;
        }
    }
    ASSERT_GE(compared, std::min(seq0.q_len, kMinComparedTokens));

    ASSERT_GE(margins1[0], kMinSelectionMargin);
    const size_t base = seq0.q_len * width;
    for (size_t i = 0; i < width; i++) {
        ASSERT_NEAR(got[base + i], expected1[i], 8e-2f) << "seq1 (decode) mismatch at elem " << i;
    }
}

// --- production checkpoint shape -------------------------------------------

TEST(qsa_gpu_test, production_shape_prefill) {
    // head_dim = 256, indexer_head_dim = 128 with rotary_dim = 64, Hkv = 2 and the real
    // PagedAttention page size. This is the configuration the Qwen3.8 checkpoint runs, so
    // it is the one that decides whether the systolic paths are reachable at all.
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const auto d = prod_dims();
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 0, 60);
    auto ref = make_reference(rg, d, /*past_len=*/0, /*q_len=*/60, /*has_gate=*/true);
    expect_qsa_matches(ref, true, 8e-2);
}

TEST(qsa_gpu_test, production_shape_chunked_prefill_with_past) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const auto d = prod_dims();
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 30, 31);
    auto ref = make_reference(rg, d, /*past_len=*/30, /*q_len=*/31, /*has_gate=*/false);
    expect_qsa_matches(ref, false, 8e-2);
}

TEST(qsa_gpu_test, production_shape_decode) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const auto d = prod_dims();
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 61, 1);
    auto ref = make_reference(rg, d, /*past_len=*/61, /*q_len=*/1, /*has_gate=*/true);
    expect_qsa_matches(ref, true, 8e-2);
}

TEST(qsa_gpu_test, production_gqa_heads_prefill) {
    // n_rep = 24 / 2 = 12, so the systolic attention packs 4 heads per dpas tile and its
    // KV reads are shared by all of them; `prod_dims` only reaches a 2-head tile.
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const auto d = prod_gqa_dims(320);
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 0, 300);
    auto ref = make_reference(rg, d, /*past_len=*/0, /*q_len=*/300, /*has_gate=*/true);
    expect_qsa_matches(ref, true, 8e-2);
}

TEST(qsa_gpu_test, production_gqa_heads_chunked_prefill) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const auto d = prod_gqa_dims(320);
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 263, 37);
    auto ref = make_reference(rg, d, /*past_len=*/263, /*q_len=*/37, /*has_gate=*/false);
    expect_qsa_matches(ref, false, 8e-2);
}

// --- long contexts ---------------------------------------------------------
//
// The sequence length drives several things at once: the number of query tiles, whether
// the last tile is partial, how many PA pages a sequence spans, and - most importantly -
// how sparse the selection is. `indexer_budget` is 2048 tokens in the real checkpoint, so
// crossing that boundary separates "the top-k keeps almost everything" from "the top-k
// keeps a small fraction"; the lengths below straddle it in both directions and include
// both multiples and non-multiples of the 16-query dpas tile.
constexpr size_t kLongContexts[] = {1563, 2048, 3657, 4096};

class qsa_long_prefill_test : public ::testing::TestWithParam<size_t> {};

TEST_P(qsa_long_prefill_test, production_shape_matches_reference) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const size_t q_len = GetParam();
    const auto d = prod_dims(q_len + 16);
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 0, q_len);
    auto ref = make_reference(rg, d, /*past_len=*/0, q_len, /*has_gate=*/true);
    expect_qsa_matches(ref, true, 8e-2);
}

class qsa_long_decode_test : public ::testing::TestWithParam<size_t> {};

TEST_P(qsa_long_decode_test, production_shape_matches_reference) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const size_t past_len = GetParam();
    const auto d = prod_dims(past_len + 16);
    ASSERT_SELECTION_IS_DISCRIMINATING(d, past_len, 1);
    auto ref = make_reference(rg, d, past_len, /*q_len=*/1, /*has_gate=*/true);
    expect_qsa_matches(ref, true, 8e-2);
}

/// Chunked prefill against a long history: the new chunk is a small fraction of the
/// context, so the page table is walked at a large offset and the queries of one tile no
/// longer share a selection.
class qsa_long_chunked_prefill_test : public ::testing::TestWithParam<size_t> {};

TEST_P(qsa_long_chunked_prefill_test, production_shape_matches_reference) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const size_t total = GetParam();
    const size_t q_len = 37;  // not a multiple of the 16-query dpas tile
    const size_t past_len = total - q_len;
    const auto d = prod_dims(total + 16);
    ASSERT_SELECTION_IS_DISCRIMINATING(d, past_len, q_len);
    auto ref = make_reference(rg, d, past_len, q_len, /*has_gate=*/false);
    expect_qsa_matches(ref, false, 8e-2);
}

INSTANTIATE_TEST_SUITE_P(smoke, qsa_long_prefill_test, ::testing::ValuesIn(kLongContexts));
INSTANTIATE_TEST_SUITE_P(smoke, qsa_long_decode_test, ::testing::ValuesIn(kLongContexts));
INSTANTIATE_TEST_SUITE_P(smoke, qsa_long_chunked_prefill_test, ::testing::ValuesIn(kLongContexts));

/// Production head counts on a long context. This is the configuration the systolic
/// attention is tuned for - a 4-head dpas tile over a deeply sparse selection - so it is
/// the one that has to stay correct as the tiling changes. Kept shorter than
/// `kLongContexts` because the reference cost scales with Hq = 24.
class qsa_long_gqa_prefill_test : public ::testing::TestWithParam<size_t> {};

TEST_P(qsa_long_gqa_prefill_test, matches_reference) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const size_t q_len = GetParam();
    const auto d = prod_gqa_dims(q_len + 16);
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 0, q_len);
    auto ref = make_reference(rg, d, /*past_len=*/0, q_len, /*has_gate=*/true);
    expect_qsa_matches(ref, true, 8e-2);
}

INSTANTIATE_TEST_SUITE_P(smoke, qsa_long_gqa_prefill_test, ::testing::Values(1024, 1563));

// --- attribute validation (no GPU required) --------------------------------

TEST(qsa_gpu_test, config_validation_rejects_illegal_attributes) {
    std::string reason;
    qsa_dims d;

    auto base = make_config(d, false);
    ASSERT_TRUE(ov::intel_gpu::op::QSA::validate_config(base, reason)) << reason;

    auto c = base;
    c.indexer_kv_heads = 2;  // indexer is MQA
    ASSERT_FALSE(ov::intel_gpu::op::QSA::validate_config(c, reason));

    c = base;
    c.budget = 9;  // budget % compress_ratio != 0
    ASSERT_FALSE(ov::intel_gpu::op::QSA::validate_config(c, reason));

    c = base;
    c.block_topk = 3;  // block_topk != budget / compress_ratio
    ASSERT_FALSE(ov::intel_gpu::op::QSA::validate_config(c, reason));

    c = base;
    c.indexer_rotary_dim = c.indexer_head_dim + 2;  // partial rotary must fit
    ASSERT_FALSE(ov::intel_gpu::op::QSA::validate_config(c, reason));

    c = base;
    c.qsa_block_tokens = c.compress_ratio + 1;
    ASSERT_FALSE(ov::intel_gpu::op::QSA::validate_config(c, reason));

    c = base;
    c.pa_block_size = 18;  // pa_block_size % compress_ratio != 0
    ASSERT_FALSE(ov::intel_gpu::op::QSA::validate_config(c, reason));

    c = base;
    c.compress_ratio = 0;  // a non-QSA (Gated DeltaNet) layer must not become a QSA primitive
    ASSERT_FALSE(ov::intel_gpu::op::QSA::validate_config(c, reason));
}

TEST(qsa_gpu_test, max_selected_tokens_reserves_the_tail) {
    qsa_dims d;
    qsa prim("qsa", {}, make_config(d, false));
    // budget + compress_ratio - 1: budget expanded tokens plus at most r-1 tail tokens.
    ASSERT_EQ(prim.max_selected_tokens(), d.budget + d.r - 1);
}

// --- roofline perf harness (token_size sweep) ------------------------------
//
// These tests do not check the QSA output value; they build the production-shaped
// network once per (phase, token_size), execute it repeatedly with GPU-side profiling
// enabled, and print one JSON line per measurement. Feed the captured lines into
// `qsa_gr_roofline.py --measured-ms` to compare the analytical roofline bound for
// Q0..Q3 against a real device execution. QSA is a single fused primitive from clDNN's
// point of view, so the number reported here is the *combined* Q0..Q3 time; compare it
// against the *sum* of the per-stage roofline bounds for the same (phase, tokens). Run
// with (also_run_disabled is required because these are marked DISABLED_ like the other
// perf-only suites in this test tree):
//   ov_gpu_unit_tests --gtest_also_run_disabled_tests --gtest_filter='*DISABLED_roofline*'
namespace roofline {

double mean_ms(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) {
        sum += x;
    }
    return v.empty() ? 0.0 : sum / static_cast<double>(v.size());
}

/// Extracts the GPU-side "executing" duration of `ev`, in milliseconds. GPU-side timing
/// is used instead of host wall-clock so host dispatch overhead (which the roofline
/// model does not account for) is excluded from the comparison.
double executing_ms(const event::ptr& ev) {
    if (ev == nullptr) {
        return -1.0;
    }
    for (const auto& interval : ev->get_profiling_info()) {
        if (interval.stage == instrumentation::profiling_stage::executing && interval.value != nullptr) {
            return static_cast<double>(interval.value->value().count()) / 1.0e6;
        }
    }
    return -1.0;
}

/// Executes `net` `iterations` times (after `warmup` untimed iterations) and returns the
/// mean GPU-side execution time of every *actually dispatched* CM kernel, keyed by the
/// stage name recorded via qsa_stage_events.hpp. QSA fuses several independent kernel
/// dispatches (Q0 kv cache update, Q1 prepare, Q2 score, Q2 top-k finalization, Q3
/// attention) behind one clDNN primitive, so `network::get_executed_primitives()` alone
/// only exposes the last of them.
///
/// `outputs.at("qsa").get_memory()` only forces a *data* dependency wait on the primitive's
/// own returned event (Q3's), which transitively depends on every earlier stage - so by the
/// time it returns, all GPU work for this call is done. But the intermediate stage events
/// themselves are never individually waited on or queried anywhere else, so each one is
/// explicitly `wait()`-ed before being read/discarded here: this is a no-op if the driver
/// already reaped it via the transitive dependency, but makes sure no retained event object
/// outlives this iteration without ever being explicitly synchronized on.
std::map<std::string, std::vector<double>> time_qsa_stages(network& net, int warmup, int iterations) {
    for (int i = 0; i < warmup; i++) {
        auto outputs = net.execute();
        outputs.at("qsa").get_memory();
        for (auto& stage_event : ov::intel_gpu::cm::qsa_debug::take_stage_events()) {
            if (stage_event.second != nullptr) {
                stage_event.second->wait();
            }
        }
    }
    std::map<std::string, std::vector<double>> samples;
    for (int i = 0; i < iterations; i++) {
        auto outputs = net.execute();
        outputs.at("qsa").get_memory();  // forces the whole call chain (Q0..Q3) to complete
        for (const auto& stage_event : ov::intel_gpu::cm::qsa_debug::take_stage_events()) {
            if (stage_event.second != nullptr) {
                stage_event.second->wait();
            }
            const double ms = executing_ms(stage_event.second);
            if (ms >= 0.0) {
                samples[stage_event.first].push_back(ms);
            }
        }
    }
    return samples;
}

void print_measurement(const char* component, const char* phase, size_t tokens, const std::string& stage, double ms, size_t samples) {
    std::cout << "ROOFLINE_JSON {\"component\":\"" << component << "\",\"phase\":\"" << phase
               << "\",\"tokens\":" << tokens << ",\"stage\":\"" << stage << "\",\"ms\":" << ms
               << ",\"samples\":" << samples << "}" << std::endl;
}

/// Builds and times the production-shaped QSA network for one (phase, token_size) point.
/// `phase == "prefill"` submits the whole context as one call (`q_len = tokens`);
/// `phase == "decode"` submits a single token after `tokens - 1` cached ones. Unlike
/// `make_reference()`, this always calls `make_reference_once()` directly: the decode
/// retry-for-discriminating-margin loop exists only for correctness tests and would be
/// wasted CPU time here, since this harness never checks the attention output.
void run_point(const char* phase, size_t tokens) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const bool decode = std::string(phase) == "decode";
    const size_t past_len = decode ? tokens - 1 : 0;
    const size_t q_len = decode ? 1 : tokens;

    // prod_gqa_dims() shrinks hidden_size and the indexer budget to bound the CPU
    // reference cost of the *correctness* tests; restore the real production values
    // (design section 3) here so the measurement matches qsa_gr_roofline.py's defaults.
    qsa_dims d = prod_gqa_dims(tokens + 16);
    d.D = 2560;
    d.budget = 2048;
    d.block_topk = 512;

    auto ref = make_reference_once(rg, d, past_len, q_len, /*has_gate=*/false);

    auto& engine = get_test_engine();
    const auto config_attrs = make_config(d, false);
    const size_t total = ref.total_len();
    const size_t num_pages = (total + d.pa_block - 1) / d.pa_block;
    const size_t summaries_per_page = d.pa_block / d.r;

    const auto slice = [&](const std::vector<float>& src, size_t stride) {
        return std::vector<float>(src.begin() + ref.past_len * stride, src.begin() + total * stride);
    };
    const auto cur_mixed = slice(ref.mixed_all, d.D);
    const auto cur_key = slice(ref.key_all, d.Hkv * d.Dh);
    const auto cur_value = slice(ref.value_all, d.Hkv * d.Dv);

    std::vector<float> key_cache(num_pages * d.Hkv * d.pa_block * d.Dh, 0.0f);
    std::vector<float> value_cache(num_pages * d.Hkv * d.pa_block * d.Dv, 0.0f);
    std::vector<float> raw_k_cache(num_pages * d.pa_block * d.Di, 0.0f);
    std::vector<float> summary_cache(num_pages * summaries_per_page * d.Di, 0.0f);
    for (size_t p = 0; p < ref.past_len; p++) {
        const size_t page = p / d.pa_block;
        const size_t slot = p % d.pa_block;
        for (size_t h = 0; h < d.Hkv; h++) {
            for (size_t j = 0; j < d.Dh; j++) {
                key_cache[((page * d.Hkv + h) * d.pa_block + slot) * d.Dh + j] = ref.key_all[p * d.Hkv * d.Dh + h * d.Dh + j];
            }
            for (size_t j = 0; j < d.Dv; j++) {
                value_cache[((page * d.Hkv + h) * d.pa_block + slot) * d.Dv + j] = ref.value_all[p * d.Hkv * d.Dv + h * d.Dv + j];
            }
        }
        const auto proj = ref.project(p);
        for (size_t j = 0; j < d.Di; j++) {
            raw_k_cache[(page * d.pa_block + slot) * d.Di + j] = proj[d.Hidx * d.Di + j];
        }
    }
    for (size_t b = 0; b * d.r + d.r <= ref.past_len; b++) {
        const auto s = ref.summary(b);
        const size_t page = b / summaries_per_page;
        const size_t slot = b % summaries_per_page;
        for (size_t j = 0; j < d.Di; j++) {
            summary_cache[(page * summaries_per_page + slot) * d.Di + j] = s[j];
        }
    }

    std::vector<int32_t> block_indices(num_pages);
    std::iota(block_indices.begin(), block_indices.end(), 0);
    const std::vector<int32_t> block_indices_begins = {0, static_cast<int32_t>(num_pages)};
    const std::vector<int32_t> past_lens = {static_cast<int32_t>(ref.past_len)};
    const std::vector<int32_t> subsequence_begins = {0, static_cast<int32_t>(ref.q_len)};
    std::vector<int32_t> position_ids(ref.q_len);
    for (size_t t = 0; t < ref.q_len; t++) {
        position_ids[t] = static_cast<int32_t>(ref.past_len + t);
    }

    const size_t J = (d.Hidx + 1) * d.Di;

    topology topology;
    topology.add(input_layout("mixed", layout{{static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.D)}, data_types::f16, format::bfyx}));
    topology.add(input_layout("query", layout{{static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.Hq * d.Dh)}, data_types::f16, format::bfyx}));
    topology.add(input_layout("key", layout{{static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.Hkv * d.Dh)}, data_types::f16, format::bfyx}));
    topology.add(input_layout("value", layout{{static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.Hkv * d.Dv)}, data_types::f16, format::bfyx}));
    topology.add(data("past_lens", make_i32(engine, {1}, past_lens)));
    topology.add(data("subsequence_begins", make_i32(engine, {2}, subsequence_begins)));
    topology.add(data("block_indices", make_i32(engine, {static_cast<int64_t>(num_pages)}, block_indices)));
    topology.add(data("block_indices_begins", make_i32(engine, {2}, block_indices_begins)));
    topology.add(input_layout("key_cache",
                              layout{{static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block),
                                      static_cast<int64_t>(d.Dh)},
                                     data_types::f16,
                                     format::bfyx}));
    topology.add(input_layout("value_cache",
                              layout{{static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block),
                                      static_cast<int64_t>(d.Dv)},
                                     data_types::f16,
                                     format::bfyx}));
    topology.add(input_layout("raw_k_cache",
                              layout{{static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Di)},
                                     data_types::f16,
                                     format::bfyx}));
    topology.add(input_layout("summary_cache",
                              layout{{static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)},
                                     data_types::f16,
                                     format::bfyx}));
    topology.add(data("idx_w", make_f16(engine, {static_cast<int64_t>(J), static_cast<int64_t>(d.D)}, ref.idx_w)));
    topology.add(data("q_norm", make_f16(engine, {static_cast<int64_t>(d.Di)}, ref.q_norm)));
    topology.add(data("k_norm", make_f16(engine, {static_cast<int64_t>(d.Di)}, ref.k_norm)));
    topology.add(data("rotary", make_f16(engine, {static_cast<int64_t>(d.max_positions), static_cast<int64_t>(d.rotary)}, ref.rotary)));
    topology.add(data("position_ids", make_i32(engine, {static_cast<int64_t>(ref.q_len)}, position_ids)));

    const std::vector<input_info> inputs = {input_info("mixed"),
                                            input_info("query"),
                                            input_info("key"),
                                            input_info("value"),
                                            input_info("past_lens"),
                                            input_info("subsequence_begins"),
                                            input_info("block_indices"),
                                            input_info("block_indices_begins"),
                                            input_info("key_cache"),
                                            input_info("value_cache"),
                                            input_info("raw_k_cache"),
                                            input_info("summary_cache"),
                                            input_info("idx_w"),
                                            input_info("q_norm"),
                                            input_info("k_norm"),
                                            input_info("rotary"),
                                            input_info("position_ids")};
    static_assert(QSAIn::COUNT == 17, "keep the test input order in sync with QSA::InputIdx");
    topology.add(qsa("qsa", inputs, config_attrs));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::optimize_data(true));
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cfg.set_property(ov::enable_profiling(true));
    network net(engine, topology, cfg);

    net.set_input_data("mixed", make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.D)}, cur_mixed));
    net.set_input_data("query", make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.Hq * d.Dh)}, ref.query_cur));
    net.set_input_data("key", make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.Hkv * d.Dh)}, cur_key));
    net.set_input_data("value", make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.Hkv * d.Dv)}, cur_value));
    net.set_input_data("key_cache",
                       make_f16(engine,
                                {static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block),
                                 static_cast<int64_t>(d.Dh)},
                                key_cache));
    net.set_input_data("value_cache",
                       make_f16(engine,
                                {static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block),
                                 static_cast<int64_t>(d.Dv)},
                                value_cache));
    net.set_input_data("raw_k_cache",
                       make_f16(engine, {static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Di)}, raw_k_cache));
    net.set_input_data("summary_cache",
                       make_f16(engine,
                                {static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)},
                                summary_cache));

    const auto stage_samples = time_qsa_stages(net, /*warmup=*/2, /*iterations=*/100);
    for (const auto& [stage, samples] : stage_samples) {
        print_measurement("QSA", phase, tokens, stage, mean_ms(samples), samples.size());
    }
}

}  // namespace roofline

struct qsa_roofline_case {
    size_t tokens;
    bool decode;
};

class qsa_roofline_perf : public ::testing::TestWithParam<qsa_roofline_case> {};

TEST_P(qsa_roofline_perf, measures_cm_kernel_time) {
    const auto p = GetParam();
    roofline::run_point(p.decode ? "decode" : "prefill", p.tokens);
}

INSTANTIATE_TEST_SUITE_P(DISABLED_roofline,
                         qsa_roofline_perf,
                         ::testing::Values(qsa_roofline_case{1024, false},
                                           qsa_roofline_case{1024, true},
                                           qsa_roofline_case{2048, false},
                                           qsa_roofline_case{2048, true},
                                           qsa_roofline_case{4096, false},
                                           qsa_roofline_case{4096, true}));

}  // namespace
