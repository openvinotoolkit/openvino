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
#include <intel_gpu/op/qsa_indexer_cache.hpp>
#include <intel_gpu/op/qsa_sdpa.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/paged_attention.hpp>
#include <intel_gpu/primitives/qsa.hpp>
#include <intel_gpu/runtime/internal_properties.hpp>
#include <intel_gpu/runtime/profiling.hpp>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/runtime/core.hpp>
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
#define ASSERT_SELECTION_IS_DISCRIMINATING(d, past_len, q_len)                                                             \
    do {                                                                                                                   \
        const size_t n_complete_ = ((past_len) + (q_len)) / (d).r;                                                         \
        ASSERT_GE(n_complete_, 3 * (d).block_topk) << "the top-k selects " << (d).block_topk << " of only " << n_complete_ \
                                                   << " blocks, so a wrong selection cannot change the output";            \
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
void apply_rope(std::vector<float>& v, const std::vector<float>& cos_full, const std::vector<float>& sin_full, size_t rotary, size_t position) {
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
///
/// `skip_kv_write` simulates the QSASDPA dense/degenerate-paging lowering: when true, the
/// main K/V cache is pre-populated for the FULL history INCLUDING this call's new tokens
/// (as if an external ov::intel_gpu::op::KVCache node had already appended them before this
/// primitive runs), Q0 (qsa_kv_cache_update) is never dispatched, and the raw `key`/`value`
/// current-step inputs are unused (still supplied, but their content is irrelevant). Q1's
/// own raw-K/summary cache extension is unaffected either way: it always runs and always
/// only pre-populates up to `past_len`, then appends the new tokens itself.
std::vector<float> run_qsa(const qsa_reference& ref, bool has_gate, bool skip_kv_write = false) {
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

    // Caches: history tokens are pre-populated exactly the way Q0 / Q1 would have. When
    // skip_kv_write is set, the main K/V cache is pre-populated all the way to `total`
    // (this call's new tokens included) instead of stopping at `past_len`, since nothing
    // will write them during execute() -- exactly what an external KVCache node's Concat
    // output would already contain.
    std::vector<float> key_cache(num_pages * d.Hkv * d.pa_block * d.Dh, 0.0f);
    std::vector<float> value_cache(num_pages * d.Hkv * d.pa_block * d.Dv, 0.0f);
    std::vector<float> raw_k_cache(num_pages * d.pa_block * d.Di, 0.0f);
    std::vector<float> summary_cache(num_pages * summaries_per_page * d.Di, 0.0f);

    const size_t kv_prefilled = skip_kv_write ? total : ref.past_len;
    for (size_t p = 0; p < kv_prefilled; p++) {
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
    }
    for (size_t p = 0; p < ref.past_len; p++) {
        const size_t page = p / d.pa_block;
        const size_t slot = p % d.pa_block;
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
    topology.add(
        input_layout("key_cache",
                     layout{{static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dh)},
                            data_types::f16,
                            format::bfyx}));
    topology.add(
        input_layout("value_cache",
                     layout{{static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dv)},
                            data_types::f16,
                            format::bfyx}));
    topology.add(input_layout(
        "raw_k_cache",
        layout{{static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Di)}, data_types::f16, format::bfyx}));
    topology.add(input_layout(
        "summary_cache",
        layout{{static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)}, data_types::f16, format::bfyx}));
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

    topology.add(qsa("qsa", inputs, config_attrs, skip_kv_write));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::optimize_data(true));
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, cfg);

    network.set_input_data("mixed", make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.D)}, cur_mixed));
    network.set_input_data("query", make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(q_width)}, ref.query_cur));
    network.set_input_data("key", make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.Hkv * d.Dh)}, cur_key));
    network.set_input_data("value", make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.Hkv * d.Dv)}, cur_value));
    network.set_input_data(
        "key_cache",
        make_f16(engine,
                 {static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dh)},
                 key_cache));
    network.set_input_data(
        "value_cache",
        make_f16(engine,
                 {static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dv)},
                 value_cache));
    network.set_input_data("raw_k_cache",
                           make_f16(engine, {static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Di)}, raw_k_cache));
    network.set_input_data(
        "summary_cache",
        make_f16(engine, {static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)}, summary_cache));

    auto outputs = network.execute();
    cldnn::mem_lock<ov::float16> out(outputs.at("qsa").get_memory(), get_test_stream());
    std::vector<float> result(out.size());
    for (size_t i = 0; i < out.size(); i++) {
        result[i] = static_cast<float>(out[i]);
    }
    return result;
}

/// Read-back of the two persistent indexer cache buffers after a `prepare_only` run.
struct qsa_prepare_only_result {
    std::vector<float> raw_k_cache;
    std::vector<float> summary_cache;
    size_t pa_block = 0;
    size_t summaries_per_page = 0;
};

/// Runs QSA with `prepare_only=true` (see cldnn::qsa::prepare_only): only Q1
/// (qsa_prepare / qsa_project_dpas) executes, updating the raw-K/summary cache buffers
/// in place exactly as it already does inside a normal QSA call -- Q0/Q2/Q3 never run,
/// and the primitive's declared attention output is left uninitialized (not read here).
///
/// This is the primitive-level foundation of the future `QSAIndexerCache` op (see
/// QSA_SDPA_DESIGN.md "Option 2"): that op would wrap this exact call pattern behind a
/// `VariableExtension`, so verifying here that (a) Q1 alone correctly extends raw-K/
/// summary from an arbitrary partially-populated starting point, and (b) doing so is
/// numerically IDENTICAL to what a normal (non-prepare_only) call already produces, is
/// exactly the load-bearing assumption that op will depend on.
///
/// The main K/V cache inputs are irrelevant in this mode (Q0/Q3 never touch them) and are
/// left zero-filled; `query`/`key`/`value` are likewise zero-filled since Q1 never reads
/// them (it only reads `mixed`).
qsa_prepare_only_result run_qsa_prepare_only(const qsa_reference& ref, bool has_gate) {
    auto& engine = get_test_engine();
    const auto& d = ref.d;
    const auto config_attrs = make_config(d, has_gate);

    const size_t total = ref.total_len();
    const size_t num_pages = (total + d.pa_block - 1) / d.pa_block;
    const size_t summaries_per_page = d.pa_block / d.r;

    const auto cur_mixed = std::vector<float>(ref.mixed_all.begin() + ref.past_len * d.D, ref.mixed_all.begin() + total * d.D);

    std::vector<float> key_cache(num_pages * d.Hkv * d.pa_block * d.Dh, 0.0f);
    std::vector<float> value_cache(num_pages * d.Hkv * d.pa_block * d.Dv, 0.0f);
    std::vector<float> raw_k_cache(num_pages * d.pa_block * d.Di, 0.0f);
    std::vector<float> summary_cache(num_pages * summaries_per_page * d.Di, 0.0f);

    // Only the already-cached HISTORY is pre-populated; Q1 must append/pool this call's
    // new tokens itself, exactly like the normal (non-prepare_only) path already does.
    for (size_t p = 0; p < ref.past_len; p++) {
        const size_t page = p / d.pa_block;
        const size_t slot = p % d.pa_block;
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
    topology.add(
        input_layout("key_cache",
                     layout{{static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dh)},
                            data_types::f16,
                            format::bfyx}));
    topology.add(
        input_layout("value_cache",
                     layout{{static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dv)},
                            data_types::f16,
                            format::bfyx}));
    topology.add(input_layout(
        "raw_k_cache",
        layout{{static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Di)}, data_types::f16, format::bfyx}));
    topology.add(input_layout(
        "summary_cache",
        layout{{static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)}, data_types::f16, format::bfyx}));
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

    topology.add(qsa("qsa", inputs, config_attrs, /*skip_kv_write=*/false, /*prepare_only=*/true));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::optimize_data(true));
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, cfg);

    network.set_input_data("mixed", make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.D)}, cur_mixed));
    network.set_input_data("query",
                           make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(q_width)}, std::vector<float>(ref.q_len * q_width, 0.0f)));
    network.set_input_data(
        "key",
        make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.Hkv * d.Dh)}, std::vector<float>(ref.q_len * d.Hkv * d.Dh, 0.0f)));
    network.set_input_data(
        "value",
        make_f16(engine, {static_cast<int64_t>(ref.q_len), static_cast<int64_t>(d.Hkv * d.Dv)}, std::vector<float>(ref.q_len * d.Hkv * d.Dv, 0.0f)));
    network.set_input_data(
        "key_cache",
        make_f16(engine,
                 {static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dh)},
                 key_cache));
    network.set_input_data(
        "value_cache",
        make_f16(engine,
                 {static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dv)},
                 value_cache));
    auto raw_k_mem = make_f16(engine, {static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Di)}, raw_k_cache);
    network.set_input_data("raw_k_cache", raw_k_mem);
    auto summary_mem =
        make_f16(engine, {static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)}, summary_cache);
    network.set_input_data("summary_cache", summary_mem);

    network.execute();

    qsa_prepare_only_result result;
    result.pa_block = d.pa_block;
    result.summaries_per_page = summaries_per_page;
    {
        cldnn::mem_lock<ov::float16> lock(raw_k_mem, get_test_stream());
        result.raw_k_cache.resize(lock.size());
        for (size_t i = 0; i < lock.size(); i++) {
            result.raw_k_cache[i] = static_cast<float>(lock[i]);
        }
    }
    {
        cldnn::mem_lock<ov::float16> lock(summary_mem, get_test_stream());
        result.summary_cache.resize(lock.size());
        for (size_t i = 0; i < lock.size(); i++) {
            result.summary_cache[i] = static_cast<float>(lock[i]);
        }
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
    topology.add(
        input_layout("key_cache",
                     layout{{static_cast<int64_t>(total_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dh)},
                            data_types::f16,
                            format::bfyx}));
    topology.add(
        input_layout("value_cache",
                     layout{{static_cast<int64_t>(total_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dv)},
                            data_types::f16,
                            format::bfyx}));
    topology.add(input_layout(
        "raw_k_cache",
        layout{{static_cast<int64_t>(total_pages), 1, static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Di)}, data_types::f16, format::bfyx}));
    topology.add(input_layout(
        "summary_cache",
        layout{{static_cast<int64_t>(total_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)}, data_types::f16, format::bfyx}));
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
    network.set_input_data(
        "key_cache",
        make_f16(engine,
                 {static_cast<int64_t>(total_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dh)},
                 key_cache));
    network.set_input_data(
        "value_cache",
        make_f16(engine,
                 {static_cast<int64_t>(total_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dv)},
                 value_cache));
    network.set_input_data("raw_k_cache",
                           make_f16(engine, {static_cast<int64_t>(total_pages), 1, static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Di)}, raw_k_cache));
    network.set_input_data(
        "summary_cache",
        make_f16(engine, {static_cast<int64_t>(total_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)}, summary_cache));

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
void expect_qsa_matches(const qsa_reference& ref, bool has_gate, float tol, bool skip_kv_write = false) {
    const auto got = run_qsa(ref, has_gate, skip_kv_write);
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

// ---------------------------------------------------------------------------
// QSASDPA foundations: degenerate paging + skip_kv_write. These do not exercise any new
// production op yet (that is out of scope for this pass) -- they validate, on real
// GPU, the two load-bearing assumptions the eventual QSASDPA lowering depends on:
//   1. a "degenerate" page table (one physical page per sequence, sized to cover the
//      whole current history) is numerically indistinguishable from an ordinary
//      small-page configuration serving the same logical history;
//   2. skipping Q0 (skip_kv_write) while the caller has already written this call's
//      new K/V into the cache produces the exact same result as the normal path.
// ---------------------------------------------------------------------------

TEST(qsa_gpu_test, degenerate_paging_prefill_matches_reference) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    const size_t past_len = 0, q_len = 64;
    d.pa_block = 64;  // one page covers the entire [0, past_len + q_len) history
    ASSERT_EQ((past_len + q_len + d.pa_block - 1) / d.pa_block, 1u) << "test setup bug: history must fit in exactly one page";
    ASSERT_SELECTION_IS_DISCRIMINATING(d, past_len, q_len);
    auto ref = make_reference(rg, d, past_len, q_len, /*has_gate=*/false);
    expect_qsa_matches(ref, false, 5e-2);
}

TEST(qsa_gpu_test, degenerate_paging_prefill_continuation_matches_reference) {
    // Same as above but with past_len > 0 (a chunked-prefill continuation), so the single
    // degenerate page holds BOTH already-cached history and this call's new tokens.
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    const size_t past_len = 50, q_len = 14;
    d.pa_block = 64;
    ASSERT_EQ((past_len + q_len + d.pa_block - 1) / d.pa_block, 1u) << "test setup bug: history must fit in exactly one page";
    ASSERT_SELECTION_IS_DISCRIMINATING(d, past_len, q_len);
    auto ref = make_reference(rg, d, past_len, q_len, /*has_gate=*/false);
    expect_qsa_matches(ref, false, 5e-2);
}

TEST(qsa_gpu_test, degenerate_paging_decode_matches_reference) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    const size_t past_len = 63, q_len = 1;
    d.pa_block = 64;
    ASSERT_EQ((past_len + q_len + d.pa_block - 1) / d.pa_block, 1u) << "test setup bug: history must fit in exactly one page";
    ASSERT_SELECTION_IS_DISCRIMINATING(d, past_len, q_len);
    auto ref = make_reference(rg, d, past_len, q_len, /*has_gate=*/false);
    expect_qsa_matches(ref, false, 5e-2);
}

TEST(qsa_gpu_test, skip_kv_write_matches_normal_path_on_degenerate_paging) {
    // Simulates the QSASDPA lowering end to end: the main K/V cache is pre-populated for
    // the FULL history (as an external KVCache node's Concat output already would be),
    // pa_block_size is degenerate (one page for the whole history), and Q0 is skipped
    // (skip_kv_write=true). Output must be bit-for-bit identical to the normal path
    // serving the same logical history, since nothing about Q1/Q2/Q3's own math changes.
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    const size_t past_len = 50, q_len = 14;
    d.pa_block = 64;
    ASSERT_EQ((past_len + q_len + d.pa_block - 1) / d.pa_block, 1u) << "test setup bug: history must fit in exactly one page";
    ASSERT_SELECTION_IS_DISCRIMINATING(d, past_len, q_len);
    auto ref = make_reference(rg, d, past_len, q_len, /*has_gate=*/false);

    const auto normal = run_qsa(ref, false, /*skip_kv_write=*/false);
    const auto skipped = run_qsa(ref, false, /*skip_kv_write=*/true);
    expect_close(skipped, normal, 0.0f);
    expect_qsa_matches(ref, false, 5e-2, /*skip_kv_write=*/true);
}

TEST(qsa_gpu_test, skip_kv_write_matches_normal_path_on_decode) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    ASSERT_SELECTION_IS_DISCRIMINATING(d, 41, 1);
    auto ref = make_reference(rg, d, /*past_len=*/41, /*q_len=*/1, /*has_gate=*/false);

    const auto normal = run_qsa(ref, false, /*skip_kv_write=*/false);
    const auto skipped = run_qsa(ref, false, /*skip_kv_write=*/true);
    expect_close(skipped, normal, 0.0f);
}

// ---------------------------------------------------------------------------
// QSAIndexerCache foundation: prepare_only wraps Q1
// unchanged to own the indexer raw-K/summary state independently of the main
// attention side. These validate, on real GPU, that Q1 alone -- run through the
// SAME degenerate-paging addressing as the main-attention side -- extends an
// arbitrary partially-populated raw-K/summary cache exactly like the reference.
// ---------------------------------------------------------------------------

void expect_prepare_only_matches_reference(const qsa_reference& ref, const qsa_prepare_only_result& got) {
    const auto& d = ref.d;
    const size_t total = ref.total_len();
    for (size_t p = 0; p < total; p++) {
        const size_t page = p / d.pa_block;
        const size_t slot = p % d.pa_block;
        const auto proj = ref.project(p);
        for (size_t j = 0; j < d.Di; j++) {
            const float expected = proj[d.Hidx * d.Di + j];
            const float actual = got.raw_k_cache[(page * d.pa_block + slot) * d.Di + j];
            ASSERT_NEAR(actual, expected, 5e-2f) << "raw_k_cache mismatch at token " << p << ", dim " << j;
        }
    }
    const size_t n_complete = total / d.r;
    for (size_t b = 0; b < n_complete; b++) {
        const auto s = ref.summary(b);
        const size_t page = b / got.summaries_per_page;
        const size_t slot = b % got.summaries_per_page;
        for (size_t j = 0; j < d.Di; j++) {
            const float expected = s[j];
            const float actual = got.summary_cache[(page * got.summaries_per_page + slot) * d.Di + j];
            ASSERT_NEAR(actual, expected, 5e-2f) << "summary_cache mismatch at block " << b << ", dim " << j;
        }
    }
}

TEST(qsa_gpu_test, prepare_only_extends_raw_k_and_summary_from_scratch) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    auto ref = make_reference(rg, d, /*past_len=*/0, /*q_len=*/40, /*has_gate=*/false);
    const auto got = run_qsa_prepare_only(ref, false);
    expect_prepare_only_matches_reference(ref, got);
}

TEST(qsa_gpu_test, prepare_only_extends_raw_k_and_summary_from_partial_history) {
    // past_len is not a multiple of r, so the call must both append raw-K for the
    // tokens still in the current incomplete block AND pool a fresh summary once that
    // block completes -- the same two code paths decode_completes_a_block exercises
    // through the normal (attention-included) path.
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    auto ref = make_reference(rg, d, /*past_len=*/41, /*q_len=*/23, /*has_gate=*/false);
    const auto got = run_qsa_prepare_only(ref, false);
    expect_prepare_only_matches_reference(ref, got);
}

TEST(qsa_gpu_test, prepare_only_matches_reference_with_degenerate_paging) {
    // Same as above but pa_block_size covers the whole history in one page (identity
    // block_indices) -- the exact addressing a future QSAIndexerCache would use.
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    const size_t past_len = 50, q_len = 14;
    d.pa_block = 64;
    ASSERT_EQ((past_len + q_len + d.pa_block - 1) / d.pa_block, 1u) << "test setup bug: history must fit in exactly one page";
    auto ref = make_reference(rg, d, past_len, q_len, /*has_gate=*/false);
    const auto got = run_qsa_prepare_only(ref, false);
    expect_prepare_only_matches_reference(ref, got);
}

// ---------------------------------------------------------------------------
// QSASDPA op: host-side (no GPU dispatch, no ProgramBuilder lowering yet)
// construction/shape-inference tests for the new ov::intel_gpu::op::QSASDPA class. 
// It reuses ov::op::internal::QSA::Config verbatim -- only InputIdx
// (dense/KVCache-managed caches, no page table) differs from QSA 
// -- so these tests mirror qsa_gpu_test's own
// config_validation_rejects_illegal_attributes / max_selected_tokens_reserves_
// the_tail style (no cm_qsa_available() guard needed: nothing here touches the
// GPU or a cldnn primitive).
// ---------------------------------------------------------------------------

using QSASDPAIn = ov::intel_gpu::op::QSASDPA::InputIdx;

ov::OutputVector make_qsasdpa_args(const qsa_dims& d, bool has_gate) {
    static_assert(QSASDPAIn::COUNT == 13, "keep this test in sync with QSASDPA::InputIdx");
    const auto q_width = has_gate ? 2 * d.Hq * d.Dh : d.Hq * d.Dh;
    const size_t J = (d.Hidx + 1) * d.Di;

    ov::OutputVector args(QSASDPAIn::COUNT);
    args[QSASDPAIn::MIXED] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(d.D)});
    args[QSASDPAIn::QUERY] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(q_width)});
    args[QSASDPAIn::PAST_LENS] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    args[QSASDPAIn::SUBSEQUENCE_BEGINS] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    args[QSASDPAIn::KEY_CACHE_DENSE] =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(d.Hkv), -1, static_cast<int64_t>(d.Dh)});
    args[QSASDPAIn::VALUE_CACHE_DENSE] =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(d.Hkv), -1, static_cast<int64_t>(d.Dv)});
    args[QSASDPAIn::INDEXER_RAW_K_CACHE_DENSE] =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, static_cast<int64_t>(d.Di)});
    args[QSASDPAIn::INDEXER_SUMMARY_CACHE_DENSE] =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, static_cast<int64_t>(d.Di)});
    args[QSASDPAIn::INDEXER_QK_WEIGHT] =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{static_cast<int64_t>(J), static_cast<int64_t>(d.D)});
    args[QSASDPAIn::INDEXER_Q_NORM_WEIGHT] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{static_cast<int64_t>(d.Di)});
    args[QSASDPAIn::INDEXER_K_NORM_WEIGHT] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{static_cast<int64_t>(d.Di)});
    args[QSASDPAIn::ROTARY_COS_SIN] =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{static_cast<int64_t>(d.max_positions), static_cast<int64_t>(d.rotary)});
    args[QSASDPAIn::POSITION_IDS] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    return args;
}

TEST(qsa_gpu_test, qsasdpa_infers_output_shape_from_query) {
    qsa_dims d;
    const auto config = make_config(d, false);
    auto node = std::make_shared<ov::intel_gpu::op::QSASDPA>(make_qsasdpa_args(d, false), config);

    ASSERT_EQ(node->get_output_size(), 1u);
    const auto& out_ps = node->get_output_partial_shape(0);
    ASSERT_TRUE(out_ps.rank().is_static());
    ASSERT_EQ(out_ps.size(), 2u);
    ASSERT_TRUE(out_ps[0].is_dynamic());  // tokens: not known until the QUERY input's shape is
    ASSERT_EQ(out_ps[1].get_length(), static_cast<int64_t>(d.Hq * d.Dv));
    ASSERT_EQ(node->get_output_element_type(0), ov::element::f16);
}

TEST(qsa_gpu_test, qsasdpa_rejects_wrong_input_count) {
    qsa_dims d;
    const auto config = make_config(d, false);
    auto args = make_qsasdpa_args(d, false);
    args.pop_back();  // drop POSITION_IDS -> only 12 of the required 13 inputs
    ASSERT_THROW(std::make_shared<ov::intel_gpu::op::QSASDPA>(args, config), ov::Exception);
}

TEST(qsa_gpu_test, qsasdpa_clone_keeps_gpu_opset_type) {
    qsa_dims d;
    const auto config = make_config(d, false);
    auto node = std::make_shared<ov::intel_gpu::op::QSASDPA>(make_qsasdpa_args(d, false), config);
    auto clone = node->clone_with_new_inputs(node->input_values());
    ASSERT_NE(std::dynamic_pointer_cast<ov::intel_gpu::op::QSASDPA>(clone), nullptr)
        << "clone_with_new_inputs() must not silently downgrade to the base ov::op::internal::QSASDPA type";
}

TEST(qsa_gpu_test, qsasdpa_reuses_qsa_config_validation) {
    // QSASDPA::Config IS ov::op::internal::QSA::Config (verbatim reuse, see qsa_sdpa.hpp),
    // so its geometry validation must be identical to QSA's -- no separate rules to drift.
    qsa_dims d;
    auto config = make_config(d, false);
    std::string reason;
    ASSERT_TRUE(ov::intel_gpu::op::QSASDPA::validate_config(config, reason));

    config.compress_ratio = 0;  // a non-QSA (Gated DeltaNet) layer must not become a QSASDPA primitive either
    ASSERT_FALSE(ov::intel_gpu::op::QSASDPA::validate_config(config, reason));
}

// ---------------------------------------------------------------------------
// QSAIndexerCache op: host-side (no GPU dispatch, no ProgramBuilder lowering yet)
// construction/shape-inference tests for the new ov::intel_gpu::op::QSAIndexerCache
// class (design section 4.3, "Option 2"). It is a VariableExtension holding the
// raw-K Variable as its single VariableExtension-visible variable, plus a second,
// explicitly-accessed summary Variable (VariableExtension only supports one
// variable per node, see qsa_indexer_cache.hpp for the rationale).
// ---------------------------------------------------------------------------

using QSAIndexerCacheIn = ov::intel_gpu::op::QSAIndexerCache::InputIdx;

ov::OutputVector make_qsa_indexer_cache_args(const qsa_dims& d) {
    static_assert(QSAIndexerCacheIn::COUNT == 8, "keep this test in sync with QSAIndexerCache::InputIdx");
    const size_t J = (d.Hidx + 1) * d.Di;

    ov::OutputVector args(QSAIndexerCacheIn::COUNT);
    args[QSAIndexerCacheIn::MIXED] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(d.D)});
    args[QSAIndexerCacheIn::INDEXER_QK_WEIGHT] =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{static_cast<int64_t>(J), static_cast<int64_t>(d.D)});
    args[QSAIndexerCacheIn::INDEXER_Q_NORM_WEIGHT] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{static_cast<int64_t>(d.Di)});
    args[QSAIndexerCacheIn::INDEXER_K_NORM_WEIGHT] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{static_cast<int64_t>(d.Di)});
    args[QSAIndexerCacheIn::ROTARY_COS_SIN] =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{static_cast<int64_t>(d.max_positions), static_cast<int64_t>(d.rotary)});
    args[QSAIndexerCacheIn::PAST_LENS] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    args[QSAIndexerCacheIn::SUBSEQUENCE_BEGINS] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    args[QSAIndexerCacheIn::POSITION_IDS] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    return args;
}

std::shared_ptr<ov::intel_gpu::op::QSAIndexerCache> make_qsa_indexer_cache(const qsa_dims& d,
                                                                           const ov::intel_gpu::op::QSAIndexerCache::Config& config) {
    auto raw_k_variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape{-1, -1, static_cast<int64_t>(d.Di)}, ov::element::f16, "qsa_indexer_raw_k_0"});
    auto summary_variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape{-1, -1, static_cast<int64_t>(d.Di)}, ov::element::f16, "qsa_indexer_summary_0"});
    return std::make_shared<ov::intel_gpu::op::QSAIndexerCache>(make_qsa_indexer_cache_args(d), raw_k_variable, summary_variable, config);
}

TEST(qsa_gpu_test, qsa_indexer_cache_infers_output_shapes_from_variables) {
    qsa_dims d;
    const auto config = make_config(d, false);
    auto node = make_qsa_indexer_cache(d, config);

    ASSERT_EQ(node->get_output_size(), 2u);
    ASSERT_EQ(node->get_output_element_type(0), ov::element::f16);
    ASSERT_EQ(node->get_output_element_type(1), ov::element::f16);
    const auto& raw_k_ps = node->get_output_partial_shape(0);
    const auto& summary_ps = node->get_output_partial_shape(1);
    ASSERT_EQ(raw_k_ps, (ov::PartialShape{-1, -1, static_cast<int64_t>(d.Di)}));
    ASSERT_EQ(summary_ps, (ov::PartialShape{-1, -1, static_cast<int64_t>(d.Di)}));
}

TEST(qsa_gpu_test, qsa_indexer_cache_exposes_both_variable_ids) {
    qsa_dims d;
    const auto config = make_config(d, false);
    auto node = make_qsa_indexer_cache(d, config);

    ASSERT_EQ(node->get_variable_id(), "qsa_indexer_raw_k_0");
    ASSERT_EQ(node->get_summary_variable_id(), "qsa_indexer_summary_0");
}

TEST(qsa_gpu_test, qsa_indexer_cache_rejects_wrong_input_count) {
    qsa_dims d;
    const auto config = make_config(d, false);
    auto raw_k_variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape{-1, -1, static_cast<int64_t>(d.Di)}, ov::element::f16, "raw_k"});
    auto summary_variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape{-1, -1, static_cast<int64_t>(d.Di)}, ov::element::f16, "summary"});
    auto args = make_qsa_indexer_cache_args(d);
    args.pop_back();  // drop POSITION_IDS -> only 7 of the required 8 inputs
    ASSERT_THROW(std::make_shared<ov::intel_gpu::op::QSAIndexerCache>(args, raw_k_variable, summary_variable, config), ov::Exception);
}

TEST(qsa_gpu_test, qsa_indexer_cache_clone_preserves_both_variables) {
    qsa_dims d;
    const auto config = make_config(d, false);
    auto node = make_qsa_indexer_cache(d, config);
    auto clone = std::dynamic_pointer_cast<ov::intel_gpu::op::QSAIndexerCache>(node->clone_with_new_inputs(node->input_values()));

    ASSERT_NE(clone, nullptr) << "clone_with_new_inputs() must not silently downgrade the gpu_opset type";
    ASSERT_EQ(clone->get_variable_id(), node->get_variable_id());
    ASSERT_EQ(clone->get_summary_variable_id(), node->get_summary_variable_id());
}

// ---------------------------------------------------------------------------
// QSASDPA end-to-end: ov::Model -> ov::Core::compile_model("GPU") -> infer(), exercising
// the real ops/qsa_sdpa.cpp::CreateQSASDPAOp ProgramBuilder lowering (not just host-side
// shape inference like the tests above). This is the first test in the whole QSASDPA
// effort that goes through a real ov::Model and a real GPU dispatch of the lowered
// cldnn::qsa(skip_kv_write=true) primitive.
//
// The main K/V and indexer raw-K/summary caches are supplied as PLAIN, dense
// ov::op::v0::Parameter inputs -- i.e. this test exercises the "advanced caller manages
// all cache state externally" usage mode (a real, intentionally-supported mode: nothing
// requires an external KVCache/QSAIndexerCache node to feed QSASDPA, see QSA_SDPA_DESIGN.md
// section 4.2), NOT the KVCache-fused or QSAIndexerCache-backed path (QSASDPAFusion and
// QSAIndexerCache's own ProgramBuilder lowering remain future work, see the design doc).
//
// Per the lowering's own static-shape requirement (see ops/qsa_sdpa.cpp's block comment:
// pa_block_size is a JIT compile-time constant, so both B and the max-sequence-length
// capacity S must be static), `config.pa_block_size` is set to the exact total sequence
// length used by this one-shot call (a "reserve exactly what this call needs" degenerate
// case of the general "reserve max_seq_len capacity upfront" v1 strategy).
// ---------------------------------------------------------------------------

std::vector<float> run_qsasdpa_ov_model(const qsa_reference& ref, bool has_gate) {
    const auto& d = ref.d;
    const int64_t total = static_cast<int64_t>(ref.total_len());
    const int64_t q_len = static_cast<int64_t>(ref.q_len);
    const int64_t Hkv = static_cast<int64_t>(d.Hkv);
    const int64_t Dh = static_cast<int64_t>(d.Dh);
    const int64_t Dv = static_cast<int64_t>(d.Dv);
    const int64_t Di = static_cast<int64_t>(d.Di);
    const int64_t r = static_cast<int64_t>(d.r);
    const int64_t summaries = total / r;
    const int64_t J = static_cast<int64_t>((d.Hidx + 1) * d.Di);
    const int64_t q_width = has_gate ? 2 * static_cast<int64_t>(d.Hq) * Dh : static_cast<int64_t>(d.Hq) * Dh;

    // Dense main K/V cache: [1, Hkv, total, Dh/Dv], pre-populated with the FULL history
    // INCLUDING this call's new tokens -- exactly what an external KVCache node's Concat
    // output (or, here, a plain caller-managed buffer) already contains before QSASDPA
    // runs; skip_kv_write=true means Q0 never writes them itself.
    std::vector<float> key_cache_dense(Hkv * total * Dh, 0.0f);
    std::vector<float> value_cache_dense(Hkv * total * Dv, 0.0f);
    for (int64_t p = 0; p < total; p++) {
        for (int64_t h = 0; h < Hkv; h++) {
            for (int64_t j = 0; j < Dh; j++) {
                key_cache_dense[(h * total + p) * Dh + j] = ref.key_all[p * Hkv * Dh + h * Dh + j];
            }
            for (int64_t j = 0; j < Dv; j++) {
                value_cache_dense[(h * total + p) * Dv + j] = ref.value_all[p * Hkv * Dv + h * Dv + j];
            }
        }
    }

    // Dense indexer raw-K/summary caches: [1, total, Di] / [1, total/r, Di], pre-populated
    // only up to past_len -- Q1 computes and appends this call's new entries itself,
    // exactly as the raw-topology prepare_only/skip_kv_write tests already verify.
    std::vector<float> raw_k_dense(total * Di, 0.0f);
    for (size_t p = 0; p < ref.past_len; p++) {
        const auto proj = ref.project(p);
        for (int64_t j = 0; j < Di; j++) {
            raw_k_dense[p * Di + j] = proj[d.Hidx * d.Di + j];
        }
    }
    std::vector<float> summary_dense(summaries * Di, 0.0f);
    for (size_t b = 0; b * d.r + d.r <= ref.past_len; b++) {
        const auto s = ref.summary(b);
        for (int64_t j = 0; j < Di; j++) {
            summary_dense[b * Di + j] = s[j];
        }
    }

    const std::vector<float> cur_mixed(ref.mixed_all.begin() + ref.past_len * d.D, ref.mixed_all.begin() + ref.total_len() * d.D);
    const std::vector<int32_t> past_lens = {static_cast<int32_t>(ref.past_len)};
    const std::vector<int32_t> subsequence_begins = {0, static_cast<int32_t>(ref.q_len)};
    std::vector<int32_t> position_ids(ref.q_len);
    for (size_t t = 0; t < ref.q_len; t++) {
        position_ids[t] = static_cast<int32_t>(ref.past_len + t);
    }

    auto mixed_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{static_cast<size_t>(q_len), d.D});
    auto query_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{static_cast<size_t>(q_len), static_cast<size_t>(q_width)});
    auto past_lens_p = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    auto subseq_p = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    auto key_cache_p =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, d.Hkv, static_cast<size_t>(total), d.Dh});
    auto value_cache_p =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, d.Hkv, static_cast<size_t>(total), d.Dv});
    auto raw_k_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, static_cast<size_t>(total), d.Di});
    auto summary_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, static_cast<size_t>(summaries), d.Di});
    auto idx_w_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{static_cast<size_t>(J), d.D});
    auto q_norm_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{d.Di});
    auto k_norm_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{d.Di});
    auto rotary_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{d.max_positions, d.rotary});
    auto position_ids_p = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{static_cast<size_t>(q_len)});

    auto config = make_config(d, has_gate);
    config.pa_block_size = static_cast<size_t>(total);

    ov::OutputVector args = {mixed_p,   query_p,      past_lens_p, subseq_p,   key_cache_p, value_cache_p, raw_k_p,
                            summary_p, idx_w_p,      q_norm_p,    k_norm_p,   rotary_p,    position_ids_p};
    static_assert(ov::intel_gpu::op::QSASDPA::InputIdx::COUNT == 13, "keep this test in sync with QSASDPA::InputIdx");
    auto qsasdpa = std::make_shared<ov::intel_gpu::op::QSASDPA>(args, config);
    auto result = std::make_shared<ov::op::v0::Result>(qsasdpa);
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{result},
        ov::ParameterVector{mixed_p, query_p, past_lens_p, subseq_p, key_cache_p, value_cache_p, raw_k_p, summary_p, idx_w_p, q_norm_p, k_norm_p, rotary_p,
                           position_ids_p});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto infer_request = compiled.create_infer_request();

    auto fill_f16 = [&](size_t idx, const std::vector<float>& data) {
        auto tensor = infer_request.get_input_tensor(idx);
        auto* dst = tensor.data<ov::float16>();
        for (size_t i = 0; i < data.size(); i++) {
            dst[i] = ov::float16(data[i]);
        }
    };
    auto fill_i32 = [&](size_t idx, const std::vector<int32_t>& data) {
        auto tensor = infer_request.get_input_tensor(idx);
        auto* dst = tensor.data<int32_t>();
        for (size_t i = 0; i < data.size(); i++) {
            dst[i] = data[i];
        }
    };
    fill_f16(0, cur_mixed);
    fill_f16(1, ref.query_cur);
    fill_i32(2, past_lens);
    fill_i32(3, subsequence_begins);
    fill_f16(4, key_cache_dense);
    fill_f16(5, value_cache_dense);
    fill_f16(6, raw_k_dense);
    fill_f16(7, summary_dense);
    fill_f16(8, ref.idx_w);
    fill_f16(9, ref.q_norm);
    fill_f16(10, ref.k_norm);
    fill_f16(11, ref.rotary);
    fill_i32(12, position_ids);

    infer_request.infer();

    auto out_tensor = infer_request.get_output_tensor(0);
    const auto* out = out_tensor.data<ov::float16>();
    std::vector<float> result_vec(out_tensor.get_size());
    for (size_t i = 0; i < result_vec.size(); i++) {
        result_vec[i] = static_cast<float>(out[i]);
    }
    return result_vec;
}

void expect_qsasdpa_matches_reference(const qsa_reference& ref, bool has_gate, float tol = 5e-2f) {
    ov::Core core;
    std::vector<std::string> available_devices;
    try {
        available_devices = core.get_available_devices();
    } catch (...) {
    }
    const bool has_gpu = std::any_of(available_devices.begin(), available_devices.end(), [](const std::string& device) {
        return device.rfind("GPU", 0) == 0;
    });
    if (!has_gpu) {
        GTEST_SKIP() << "GPU plugin is not available in ov::Core (unit test binary may lack plugin discovery).";
    }

    const auto got = run_qsasdpa_ov_model(ref, has_gate);
    const auto want = ref.attention();
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < want.size(); i++) {
        ASSERT_NEAR(got[i], want[i], tol) << "QSASDPA output mismatch at " << i;
    }
}

TEST(qsa_gpu_test, qsasdpa_ov_model_prefill_matches_reference) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    const size_t past_len = 0, q_len = 64;
    d.pa_block = past_len + q_len;  // config.pa_block_size == the exact total length this one-shot call needs
    ASSERT_SELECTION_IS_DISCRIMINATING(d, past_len, q_len);
    auto ref = make_reference(rg, d, past_len, q_len, /*has_gate=*/false);
    expect_qsasdpa_matches_reference(ref, false);
}

TEST(qsa_gpu_test, qsasdpa_ov_model_decode_continuation_matches_reference) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    const size_t past_len = 63, q_len = 1;
    d.pa_block = past_len + q_len;
    ASSERT_SELECTION_IS_DISCRIMINATING(d, past_len, q_len);
    auto ref = make_reference(rg, d, past_len, q_len, /*has_gate=*/false);
    expect_qsasdpa_matches_reference(ref, false);
}

// MIXED: a batch of >1 sequence sharing one layer's weights, some with q_len > 1
// (prefill/continuation) and some with q_len == 1 (decode), all in a single
// network.execute() call -- exercises the SAME QSAStage::MIXED host-side dispatch path
// QSA (paged) already uses, now reached through QSASDPA's dense/degenerate lowering. To
// keep this increment's scope bounded (no S-axis padding/truncation logic, see the
// design doc's "fixed max-sequence-length capacity" finding), every sequence in `refs`
// must share the SAME total_len -- KEY_CACHE_DENSE/etc.'s shared S axis then needs no
// padding for shorter sequences, which is a separate, not-yet-implemented concern.
std::vector<float> run_qsasdpa_ov_model_batch(const std::vector<qsa_reference>& refs, bool has_gate) {
    const auto& d = refs.front().d;
    const size_t num_seqs = refs.size();
    const size_t total = refs.front().total_len();
    for (const auto& ref : refs) {
        if (ref.total_len() != total) {
            ADD_FAILURE() << "run_qsasdpa_ov_model_batch requires every sequence to share the same total_len (no S-axis padding support yet)";
            return {};
        }
    }
    const int64_t S = static_cast<int64_t>(total);
    const int64_t B = static_cast<int64_t>(num_seqs);
    const int64_t Hkv = static_cast<int64_t>(d.Hkv);
    const int64_t Dh = static_cast<int64_t>(d.Dh);
    const int64_t Dv = static_cast<int64_t>(d.Dv);
    const int64_t Di = static_cast<int64_t>(d.Di);
    const int64_t r = static_cast<int64_t>(d.r);
    const int64_t summaries = S / r;
    const int64_t J = static_cast<int64_t>((d.Hidx + 1) * d.Di);
    const int64_t q_width = has_gate ? 2 * static_cast<int64_t>(d.Hq) * Dh : static_cast<int64_t>(d.Hq) * Dh;

    size_t total_tokens = 0;
    for (const auto& ref : refs) {
        total_tokens += ref.q_len;
    }

    std::vector<float> mixed_cur, query_cur;
    std::vector<float> key_cache_dense(B * Hkv * S * Dh, 0.0f);
    std::vector<float> value_cache_dense(B * Hkv * S * Dv, 0.0f);
    std::vector<float> raw_k_dense(B * S * Di, 0.0f);
    std::vector<float> summary_dense(B * summaries * Di, 0.0f);
    std::vector<int32_t> past_lens(num_seqs);
    std::vector<int32_t> subsequence_begins(num_seqs + 1, 0);
    std::vector<int32_t> position_ids;
    position_ids.reserve(total_tokens);

    for (size_t s = 0; s < num_seqs; s++) {
        const auto& ref = refs[s];
        past_lens[s] = static_cast<int32_t>(ref.past_len);
        subsequence_begins[s + 1] = subsequence_begins[s] + static_cast<int32_t>(ref.q_len);

        const auto slice = [&](const std::vector<float>& src, size_t stride) {
            return std::vector<float>(src.begin() + ref.past_len * stride, src.begin() + total * stride);
        };
        const auto cm = slice(ref.mixed_all, d.D);
        mixed_cur.insert(mixed_cur.end(), cm.begin(), cm.end());
        query_cur.insert(query_cur.end(), ref.query_cur.begin(), ref.query_cur.end());
        for (size_t t = 0; t < ref.q_len; t++) {
            position_ids.push_back(static_cast<int32_t>(ref.past_len + t));
        }

        // Full history INCLUDING this call's new tokens -- as if an external KVCache node
        // had already appended them (skip_kv_write=true).
        for (size_t p = 0; p < total; p++) {
            for (int64_t h = 0; h < Hkv; h++) {
                for (int64_t j = 0; j < Dh; j++) {
                    key_cache_dense[((s * Hkv + h) * S + p) * Dh + j] = ref.key_all[p * Hkv * Dh + h * Dh + j];
                }
                for (int64_t j = 0; j < Dv; j++) {
                    value_cache_dense[((s * Hkv + h) * S + p) * Dv + j] = ref.value_all[p * Hkv * Dv + h * Dv + j];
                }
            }
        }
        // Indexer raw-K/summary: only pre-populated up to past_len, Q1 appends the rest.
        for (size_t p = 0; p < ref.past_len; p++) {
            const auto proj = ref.project(p);
            for (int64_t j = 0; j < Di; j++) {
                raw_k_dense[(s * S + p) * Di + j] = proj[d.Hidx * d.Di + j];
            }
        }
        for (size_t b = 0; b * d.r + d.r <= ref.past_len; b++) {
            const auto sm = ref.summary(b);
            for (int64_t j = 0; j < Di; j++) {
                summary_dense[(s * summaries + b) * Di + j] = sm[j];
            }
        }
    }

    auto mixed_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{total_tokens, d.D});
    auto query_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{total_tokens, static_cast<size_t>(q_width)});
    auto past_lens_p = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{num_seqs});
    auto subseq_p = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{num_seqs + 1});
    auto key_cache_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{static_cast<size_t>(B), d.Hkv, static_cast<size_t>(S), d.Dh});
    auto value_cache_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{static_cast<size_t>(B), d.Hkv, static_cast<size_t>(S), d.Dv});
    auto raw_k_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{static_cast<size_t>(B), static_cast<size_t>(S), d.Di});
    auto summary_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{static_cast<size_t>(B), static_cast<size_t>(summaries), d.Di});
    auto idx_w_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{static_cast<size_t>(J), d.D});
    auto q_norm_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{d.Di});
    auto k_norm_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{d.Di});
    auto rotary_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{d.max_positions, d.rotary});
    auto position_ids_p = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{total_tokens});

    auto config = make_config(d, has_gate);
    config.pa_block_size = static_cast<size_t>(S);

    ov::OutputVector args = {mixed_p,   query_p,      past_lens_p, subseq_p,   key_cache_p, value_cache_p, raw_k_p,
                            summary_p, idx_w_p,      q_norm_p,    k_norm_p,   rotary_p,    position_ids_p};
    auto qsasdpa = std::make_shared<ov::intel_gpu::op::QSASDPA>(args, config);
    auto result = std::make_shared<ov::op::v0::Result>(qsasdpa);
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{result},
        ov::ParameterVector{mixed_p, query_p, past_lens_p, subseq_p, key_cache_p, value_cache_p, raw_k_p, summary_p, idx_w_p, q_norm_p, k_norm_p, rotary_p,
                           position_ids_p});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto infer_request = compiled.create_infer_request();

    auto fill_f16 = [&](size_t idx, const std::vector<float>& data) {
        auto tensor = infer_request.get_input_tensor(idx);
        auto* dst = tensor.data<ov::float16>();
        for (size_t i = 0; i < data.size(); i++) {
            dst[i] = ov::float16(data[i]);
        }
    };
    auto fill_i32 = [&](size_t idx, const std::vector<int32_t>& data) {
        auto tensor = infer_request.get_input_tensor(idx);
        auto* dst = tensor.data<int32_t>();
        for (size_t i = 0; i < data.size(); i++) {
            dst[i] = data[i];
        }
    };
    const auto& w = refs.front();  // shared layer weights
    fill_f16(0, mixed_cur);
    fill_f16(1, query_cur);
    fill_i32(2, past_lens);
    fill_i32(3, subsequence_begins);
    fill_f16(4, key_cache_dense);
    fill_f16(5, value_cache_dense);
    fill_f16(6, raw_k_dense);
    fill_f16(7, summary_dense);
    fill_f16(8, w.idx_w);
    fill_f16(9, w.q_norm);
    fill_f16(10, w.k_norm);
    fill_f16(11, w.rotary);
    fill_i32(12, position_ids);

    infer_request.infer();

    auto out_tensor = infer_request.get_output_tensor(0);
    const auto* out = out_tensor.data<ov::float16>();
    std::vector<float> result_vec(out_tensor.get_size());
    for (size_t i = 0; i < result_vec.size(); i++) {
        result_vec[i] = static_cast<float>(out[i]);
    }
    return result_vec;
}

TEST(qsa_gpu_test, qsasdpa_ov_model_mixed_batch_matches_reference) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    ov::Core core;
    std::vector<std::string> available_devices;
    try {
        available_devices = core.get_available_devices();
    } catch (...) {
    }
    const bool has_gpu = std::any_of(available_devices.begin(), available_devices.end(), [](const std::string& device) {
        return device.rfind("GPU", 0) == 0;
    });
    if (!has_gpu) {
        GTEST_SKIP() << "GPU plugin is not available in ov::Core (unit test binary may lack plugin discovery).";
    }

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    const size_t total_len = 40;
    d.pa_block = total_len;
    ASSERT_SELECTION_IS_DISCRIMINATING(d, /*past_len=*/0, /*q_len=*/total_len);

    // Two sequences sharing one layer's weights, same total_len (so the shared dense S
    // axis needs no padding): seq 0 is a from-scratch prefill, seq 1 is a single decode
    // step continuing a long-established history -- exactly QSAStage::MIXED's defining
    // scenario (some q_len > 1, some q_len == 1, in the same batched call).
    auto seq0 = make_reference(rg, d, /*past_len=*/0, /*q_len=*/total_len, /*has_gate=*/false);
    auto seq1 = make_reference_sharing_weights(rg, seq0, /*past_len=*/total_len - 1, /*q_len=*/1, /*has_gate=*/false);

    const auto got = run_qsasdpa_ov_model_batch({seq0, seq1}, false);
    const auto want0 = seq0.attention();
    const auto want1 = seq1.attention();
    ASSERT_EQ(got.size(), want0.size() + want1.size());
    for (size_t i = 0; i < want0.size(); i++) {
        ASSERT_NEAR(got[i], want0[i], 5e-2f) << "seq0 output mismatch at " << i;
    }
    for (size_t i = 0; i < want1.size(); i++) {
        ASSERT_NEAR(got[want0.size() + i], want1[i], 5e-2f) << "seq1 output mismatch at " << i;
    }
}

// Dynamic B (or an unbounded/dynamic max-sequence-length capacity) must be rejected at
// graph-compile time with a clear error, per QSA_SDPA_DESIGN.md section 7 -- not silently
// produce wrong numbers. This exercises ops/qsa_sdpa.cpp::CreateQSASDPAOp's own static-shape
// assertions (see its block comment), reached only once the op participates in a real
// compile_model("GPU") call (the host-side tests earlier in this file never lower the op).
TEST(qsa_gpu_test, qsasdpa_rejects_dynamic_key_cache_shape) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    ov::Core core;
    std::vector<std::string> available_devices;
    try {
        available_devices = core.get_available_devices();
    } catch (...) {
    }
    const bool has_gpu = std::any_of(available_devices.begin(), available_devices.end(), [](const std::string& device) {
        return device.rfind("GPU", 0) == 0;
    });
    if (!has_gpu) {
        GTEST_SKIP() << "GPU plugin is not available in ov::Core (unit test binary may lack plugin discovery).";
    }

    qsa_dims d;
    const auto config = make_config(d, false);
    const int64_t S = static_cast<int64_t>(d.pa_block);
    const int64_t J = static_cast<int64_t>((d.Hidx + 1) * d.Di);
    const auto q_width = static_cast<int64_t>(d.Hq * d.Dh);

    auto mixed_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(d.D)});
    auto query_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, q_width});
    auto past_lens_p = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});  // dynamic B: must be rejected
    auto subseq_p = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto key_cache_p = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(d.Hkv), S, static_cast<int64_t>(d.Dh)});  // dynamic B here too
    auto value_cache_p =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(d.Hkv), S, static_cast<int64_t>(d.Dv)});
    auto raw_k_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, S, static_cast<int64_t>(d.Di)});
    auto summary_p =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, S / static_cast<int64_t>(d.r), static_cast<int64_t>(d.Di)});
    auto idx_w_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{J, static_cast<int64_t>(d.D)});
    auto q_norm_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{static_cast<int64_t>(d.Di)});
    auto k_norm_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{static_cast<int64_t>(d.Di)});
    auto rotary_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                            ov::PartialShape{static_cast<int64_t>(d.max_positions), static_cast<int64_t>(d.rotary)});
    auto position_ids_p = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});

    ov::OutputVector args = {mixed_p,   query_p,      past_lens_p, subseq_p,   key_cache_p, value_cache_p, raw_k_p,
                            summary_p, idx_w_p,      q_norm_p,    k_norm_p,   rotary_p,    position_ids_p};
    auto qsasdpa = std::make_shared<ov::intel_gpu::op::QSASDPA>(args, config);
    auto result = std::make_shared<ov::op::v0::Result>(qsasdpa);
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{result},
        ov::ParameterVector{mixed_p, query_p, past_lens_p, subseq_p, key_cache_p, value_cache_p, raw_k_p, summary_p, idx_w_p, q_norm_p, k_norm_p, rotary_p,
                           position_ids_p});

    ASSERT_THROW(core.compile_model(model, "GPU"), ov::Exception)
        << "dynamic B / KEY_CACHE_DENSE shape must be rejected at compile time (QSA_SDPA_DESIGN.md section 7), not silently mis-lowered";
}

// ---------------------------------------------------------------------------
// QSAIndexerCache end-to-end: ov::Model -> ov::Core::compile_model("GPU") -> TWO
// SEPARATE ov::InferRequest::infer() calls on the SAME compiled model, exercising the
// real ops/qsa_indexer_cache.cpp::CreateQSAIndexerCacheOp ProgramBuilder lowering
// (ReadValue -> qsa(prepare_only=true) -> Assign, see that file's comments). This is the
// piece the QSASDPA end-to-end tests above deliberately did NOT cover: those pre-populate
// the "history" portion of the indexer caches by calling the FP64 reference directly on
// the host, never by actually running a prior GPU call. Here, the SECOND infer() call's
// raw-K/summary output must reflect BOTH the first call's Q1-computed history (recovered
// purely via the persisted Variable state -- nothing about the second call's inputs
// repeats the first call's tokens) and the second call's own newly-appended tokens.
// ---------------------------------------------------------------------------

std::shared_ptr<ov::Model> make_qsa_indexer_cache_model(const qsa_dims& d, int64_t B, int64_t S) {
    const int64_t Di = static_cast<int64_t>(d.Di);
    const int64_t r = static_cast<int64_t>(d.r);
    const int64_t J = static_cast<int64_t>((d.Hidx + 1) * d.Di);

    auto mixed_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(d.D)});
    auto idx_w_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{J, static_cast<int64_t>(d.D)});
    auto q_norm_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{Di});
    auto k_norm_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{Di});
    auto rotary_p =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{static_cast<int64_t>(d.max_positions), static_cast<int64_t>(d.rotary)});
    auto past_lens_p = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{B});
    auto subseq_p = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{B + 1});
    auto position_ids_p = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});

    auto raw_k_variable =
        std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{ov::PartialShape{B, S, Di}, ov::element::f16, "qsa_indexer_raw_k_e2e"});
    auto summary_variable =
        std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{ov::PartialShape{B, S / r, Di}, ov::element::f16, "qsa_indexer_summary_e2e"});

    auto config = make_config(d, false);
    config.pa_block_size = static_cast<size_t>(S);

    ov::OutputVector args = {mixed_p, idx_w_p, q_norm_p, k_norm_p, rotary_p, past_lens_p, subseq_p, position_ids_p};
    static_assert(ov::intel_gpu::op::QSAIndexerCache::InputIdx::COUNT == 8, "keep this test in sync with QSAIndexerCache::InputIdx");
    auto cache = std::make_shared<ov::intel_gpu::op::QSAIndexerCache>(args, raw_k_variable, summary_variable, config);

    auto raw_k_result = std::make_shared<ov::op::v0::Result>(cache->output(0));
    auto summary_result = std::make_shared<ov::op::v0::Result>(cache->output(1));
    return std::make_shared<ov::Model>(
        ov::ResultVector{raw_k_result, summary_result},
        ov::ParameterVector{mixed_p, idx_w_p, q_norm_p, k_norm_p, rotary_p, past_lens_p, subseq_p, position_ids_p});
}

TEST(qsa_gpu_test, qsa_indexer_cache_ov_model_persists_state_across_two_infer_calls) {
    if (!cm_qsa_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for qsa";
    }
    ov::Core core;
    std::vector<std::string> available_devices;
    try {
        available_devices = core.get_available_devices();
    } catch (...) {
    }
    const bool has_gpu = std::any_of(available_devices.begin(), available_devices.end(), [](const std::string& device) {
        return device.rfind("GPU", 0) == 0;
    });
    if (!has_gpu) {
        GTEST_SKIP() << "GPU plugin is not available in ov::Core (unit test binary may lack plugin discovery).";
    }

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    qsa_dims d;
    const size_t q_len1 = 20, q_len2 = 20;
    const size_t total = q_len1 + q_len2;  // 40, well within the reserved capacity below
    const int64_t S = 64;                 // config.pa_block_size: reserved capacity, > total
    auto ref = make_reference(rg, d, /*past_len=*/0, /*q_len=*/total, /*has_gate=*/false);

    auto model = make_qsa_indexer_cache_model(d, /*B=*/1, S);
    auto compiled = core.compile_model(model, "GPU");
    auto infer_request = compiled.create_infer_request();

    auto set_f16 = [&](const std::shared_ptr<ov::op::v0::Parameter>& param, const ov::Shape& shape, const std::vector<float>& data) {
        ov::Tensor tensor(ov::element::f16, shape);
        auto* dst = tensor.data<ov::float16>();
        for (size_t i = 0; i < data.size(); i++) {
            dst[i] = ov::float16(data[i]);
        }
        infer_request.set_tensor(param, tensor);
    };
    auto set_i32 = [&](const std::shared_ptr<ov::op::v0::Parameter>& param, const ov::Shape& shape, const std::vector<int32_t>& data) {
        ov::Tensor tensor(ov::element::i32, shape);
        auto* dst = tensor.data<int32_t>();
        for (size_t i = 0; i < data.size(); i++) {
            dst[i] = data[i];
        }
        infer_request.set_tensor(param, tensor);
    };

    // Parameter order matches make_qsa_indexer_cache_model's ParameterVector.
    const auto& params = model->get_parameters();
    auto mixed_p = params[0];
    auto idx_w_p = params[1];
    auto q_norm_p = params[2];
    auto k_norm_p = params[3];
    auto rotary_p = params[4];
    auto past_lens_p = params[5];
    auto subseq_p = params[6];
    auto position_ids_p = params[7];

    // These do not change between calls: one layer's weights.
    set_f16(idx_w_p, ov::Shape{(d.Hidx + 1) * d.Di, d.D}, ref.idx_w);
    set_f16(q_norm_p, ov::Shape{d.Di}, ref.q_norm);
    set_f16(k_norm_p, ov::Shape{d.Di}, ref.k_norm);
    set_f16(rotary_p, ov::Shape{d.max_positions, d.rotary}, ref.rotary);

    auto run_call = [&](size_t past_len, size_t q_len) {
        const std::vector<float> cur_mixed(ref.mixed_all.begin() + past_len * d.D, ref.mixed_all.begin() + (past_len + q_len) * d.D);
        std::vector<int32_t> position_ids(q_len);
        for (size_t t = 0; t < q_len; t++) {
            position_ids[t] = static_cast<int32_t>(past_len + t);
        }
        set_f16(mixed_p, ov::Shape{q_len, d.D}, cur_mixed);
        set_i32(past_lens_p, ov::Shape{1}, {static_cast<int32_t>(past_len)});
        set_i32(subseq_p, ov::Shape{2}, {0, static_cast<int32_t>(q_len)});
        set_i32(position_ids_p, ov::Shape{q_len}, position_ids);
        infer_request.infer();
    };

    // Call 1: from scratch.
    run_call(/*past_len=*/0, q_len1);
    // Call 2: continues the SAME sequence. Crucially, `run_call` only ever feeds THIS
    // call's new tokens (mixed[q_len1:total]) -- call 1's tokens are never resubmitted.
    // If the second call's output nonetheless reflects positions [0, q_len1) correctly,
    // that content can only have come from the persisted Variable state.
    run_call(/*past_len=*/q_len1, q_len2);

    auto raw_k_tensor = infer_request.get_output_tensor(0);
    auto summary_tensor = infer_request.get_output_tensor(1);
    const auto* raw_k_out = raw_k_tensor.data<ov::float16>();
    const auto* summary_out = summary_tensor.data<ov::float16>();

    for (size_t p = 0; p < total; p++) {
        const auto proj = ref.project(p);
        for (size_t j = 0; j < d.Di; j++) {
            const float expected = proj[d.Hidx * d.Di + j];
            const float actual = static_cast<float>(raw_k_out[p * d.Di + j]);
            ASSERT_NEAR(actual, expected, 5e-2f) << "raw_k mismatch at token " << p << " (past call 1's history if p < " << q_len1
                                                 << ") dim " << j;
        }
    }
    const size_t n_complete = total / d.r;
    for (size_t b = 0; b < n_complete; b++) {
        const auto s = ref.summary(b);
        for (size_t j = 0; j < d.Di; j++) {
            const float expected = s[j];
            const float actual = static_cast<float>(summary_out[b * d.Di + j]);
            ASSERT_NEAR(actual, expected, 5e-2f) << "summary mismatch at block " << b << " dim " << j;
        }
    }
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
    std::cout << "ROOFLINE_JSON {\"component\":\"" << component << "\",\"phase\":\"" << phase << "\",\"tokens\":" << tokens << ",\"stage\":\"" << stage
              << "\",\"ms\":" << ms << ",\"samples\":" << samples << "}" << std::endl;
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
    topology.add(
        input_layout("key_cache",
                     layout{{static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dh)},
                            data_types::f16,
                            format::bfyx}));
    topology.add(
        input_layout("value_cache",
                     layout{{static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dv)},
                            data_types::f16,
                            format::bfyx}));
    topology.add(input_layout(
        "raw_k_cache",
        layout{{static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Di)}, data_types::f16, format::bfyx}));
    topology.add(input_layout(
        "summary_cache",
        layout{{static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)}, data_types::f16, format::bfyx}));
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
                                {static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dh)},
                                key_cache));
    net.set_input_data("value_cache",
                       make_f16(engine,
                                {static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Dv)},
                                value_cache));
    net.set_input_data("raw_k_cache",
                       make_f16(engine, {static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(d.pa_block), static_cast<int64_t>(d.Di)}, raw_k_cache));
    net.set_input_data(
        "summary_cache",
        make_f16(engine, {static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)}, summary_cache));

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
