// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the split PagedQSAIndexer/SparsePA and QSAIndexer/SparseSDPA primitives.
// Topologies are built directly at the cldnn primitive level.
//
// Covered:
//  - Paged and dense indexer + attention pairs against a CPU reference, both with toy
//    dims and with the Qwen3.8-Flash-Next shape (config.json): H_iq=4, D_idx=128,
//    H_ik=1, r=4, budget 2048 tokens (block_topk=512), rotary 64, Hq=24, Hkv=2, Dh=256,
//    page 16. The GPU's top-k selection is validated against the reference scores with
//    an fp16 tolerance, and the attention output is then compared using that selection,
//    so near-tied top-k boundaries cannot make the test flaky.
//  - Non-sequential (mRoPE-like) positions, which exercise index_block_rope_cache.
//  - Dedicated precision/numerical-stability stress tests: large-magnitude activations,
//    a single dominant-outlier channel (stresses RMSNorm's sum-of-squares reduction),
//    and long-context accumulation (many blocks/softmax terms). The CPU reference
//    quantizes every input through the SAME f16 round-trip the GPU sees (`make_f16`),
//    so the measured error reflects kernel compute precision, not input quantization
//    noise; every comparison asserts finiteness and prints a `QSA_PRECISION` line
//    (max/mean absolute and relative error) for inspection.
//  - Host-level op validation and plugin-level rejection of unsupported configs.
//  - DISABLED_ performance tests that print per-op timings, run with:
//      ov_gpu_unit_tests --gtest_also_run_disabled_tests --gtest_filter='*qsa_ops_perf*'

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/paged_qsa_indexer.hpp>
#include <intel_gpu/primitives/qsa_indexer.hpp>
#include <intel_gpu/primitives/sparse_pa.hpp>
#include <intel_gpu/primitives/sparse_sdpa.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <openvino/op/assign.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/paged_sparse_attention.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/read_value.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/util/variable.hpp>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/runtime/core.hpp"
#include "primitive_inst.h"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

using PagedIn = ov::intel_gpu::op::PagedQSAIndexer::InputIdx;
using DenseIdxIn = ov::intel_gpu::op::QSAIndexer::InputIdx;
using SparsePAIn = cldnn::sparse_pa::InputIdx;
using SparseSDPAIn = ov::intel_gpu::op::SparseSDPA::InputIdx;

/// The QSA ops run on the OpenCL implementation, which needs the Intel subgroup extensions.
bool qsa_ops_available() {
    auto& engine = get_test_engine();
    const auto& info = engine.get_device_info();
    return info.supports_intel_subgroups && info.supports_intel_subgroups_short;
}

int64_t I(size_t v) {
    return static_cast<int64_t>(v);
}

struct split_dims {
    size_t Hiq = 2;  // index-query heads (H_ik == 1 fixed in v1)
    size_t Di = 16;  // indexer head dim
    size_t rotary = 8;
    size_t r = 4;  // key_block_size
    size_t block_topk = 2;
    size_t page_size = 16;
    size_t max_positions = 64;
    float eps = 1e-6f;

    size_t Hq = 2;
    size_t Hkv = 1;
    size_t Dh = 16;
    size_t Dv = 16;
};

/// Qwen3.8-Flash-Next (config.json text_config). The indexer reuses the attention RoPE, so
/// rotary = head_dim (256) * partial_rotary_factor (0.25); block_topk = indexer_budget / r.
split_dims qwen38_dims() {
    split_dims d;
    d.Hiq = 4;
    d.Di = 128;
    d.rotary = 64;
    d.r = 4;
    d.block_topk = 2048 / 4;
    d.page_size = 16;
    d.eps = 1e-6f;
    d.Hq = 24;
    d.Hkv = 2;
    d.Dh = 256;
    d.Dv = 256;
    return d;
}

/// Plain DPAS attention with one worker and 4 heads per tile; GQA-DPAS needs Dh % 128 == 0.
split_dims non_gqa_dpas_dims() {
    auto d = qwen38_dims();
    d.Dh = 64;
    d.Dv = 64;
    return d;
}

/// n_rep = 32 exceeds GQA-DPAS's limit of 16; plain DPAS needs 4 workers and 8 heads per tile.
split_dims wide_group_dims() {
    auto d = qwen38_dims();
    d.Hq = 32;
    d.Hkv = 1;
    return d;
}

/// n_rep = 1: plain DPAS with one head per tile.
split_dims mha_dpas_dims() {
    auto d = non_gqa_dpas_dims();
    d.Hq = 4;
    d.Hkv = 4;
    return d;
}

std::vector<ov::float16> to_f16(const std::vector<float>& src) {
    std::vector<ov::float16> dst(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst[i] = ov::float16(src[i]);
    }
    return dst;
}

/// Rounds every element through f16, in place: used so the CPU reference computes from
/// the EXACT same values the GPU reads back (every f16 input below goes through this
/// same round-trip via `make_f16`), isolating kernel compute error from input-quantization
/// noise in the precision comparisons.
void quantize_f16(std::vector<float>& v) {
    for (auto& x : v) {
        x = static_cast<float>(ov::float16(x));
    }
}

/// Device-resident like real model tensors; the default lockable allocation is usm_shared on dGPU.
template <typename T>
memory::ptr make_device(engine& eng, const ov::PartialShape& shape, data_types dt, const std::vector<T>& data) {
    auto mem = eng.allocate_memory(layout{shape, dt, format::bfyx}, allocation_type::usm_device);
    if (!data.empty()) {
        mem->copy_from(get_test_stream(), data.data(), true);
    }
    return mem;
}

memory::ptr make_f16(engine& eng, const ov::PartialShape& shape, const std::vector<float>& data) {
    return make_device(eng, shape, data_types::f16, to_f16(data));
}

memory::ptr make_i32(engine& eng, const ov::PartialShape& shape, const std::vector<int32_t>& data) {
    return make_device(eng, shape, data_types::i32, data);
}

std::vector<int32_t> read_i32(const memory::ptr& mem) {
    mem_lock<int32_t, mem_lock_type::read> lock(mem, get_test_stream());
    return {lock.begin(), lock.end()};
}

std::vector<float> read_f16(const memory::ptr& mem) {
    mem_lock<ov::float16, mem_lock_type::read> lock(mem, get_test_stream());
    std::vector<float> out(lock.size());
    for (size_t i = 0; i < lock.size(); i++) {
        out[i] = static_cast<float>(lock[i]);
    }
    return out;
}

std::vector<float> random_vec(tests::random_generator& rg, size_t count, float scale, float bias = 0.0f) {
    auto v = rg.generate_random_1d<float>(count, -1, 1, 16);
    std::vector<float> out(v.begin(), v.end());
    for (auto& x : out) {
        x = x * scale + bias;
    }
    return out;
}

/// Cheap deterministic filler for the large perf buffers.
std::vector<float> lcg_vec(size_t count, uint32_t seed, float scale = 1.0f) {
    std::vector<float> out(count);
    uint32_t s = seed * 2654435761u + 1u;
    for (auto& x : out) {
        s = s * 1664525u + 1013904223u;
        x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f) * scale;
    }
    return out;
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

/// NEOX (rotate-half) partial RoPE using the kernel's interleaved (cos_j, sin_j) table.
void apply_rope(std::vector<float>& v, const std::vector<float>& table, size_t rotary, size_t position) {
    const size_t half = rotary / 2;
    for (size_t j = 0; j < half; j++) {
        const float c = table[position * rotary + 2 * j];
        const float s = table[position * rotary + 2 * j + 1];
        const float x0 = v[j];
        const float x1 = v[j + half];
        v[j] = x0 * c - x1 * s;
        v[j + half] = x1 * c + x0 * s;
    }
}

std::vector<float> make_rotary_table(size_t max_positions, size_t rotary) {
    std::vector<float> table(max_positions * rotary, 0.0f);
    for (size_t p = 0; p < max_positions; p++) {
        for (size_t j = 0; j < rotary / 2; j++) {
            const float freq = 1.0f / std::pow(10000.0f, static_cast<float>(2 * j) / static_cast<float>(rotary));
            const float angle = static_cast<float>(p) * freq;
            table[p * rotary + 2 * j] = std::cos(angle);
            table[p * rotary + 2 * j + 1] = std::sin(angle);
        }
    }
    return table;
}

/// The op's per-token `rope_cos`/`rope_sin` rows ([n, rotary] each, Hugging Face
/// `cat(freqs, freqs)` layout): what the producer gathers from the interleaved `table` at
/// each token's `positions` entry.
std::pair<std::vector<float>, std::vector<float>> rope_rows(const std::vector<float>& table, size_t rotary, const std::vector<int32_t>& positions) {
    const size_t half = rotary / 2;
    std::vector<float> cos(positions.size() * rotary), sin(positions.size() * rotary);
    for (size_t t = 0; t < positions.size(); t++) {
        const size_t row = static_cast<size_t>(positions[t]) * rotary;
        for (size_t j = 0; j < rotary; j++) {
            cos[t * rotary + j] = table[row + 2 * (j % half)];
            sin[t * rotary + j] = table[row + 2 * (j % half) + 1];
        }
    }
    return {cos, sin};
}

/// One RoPE state/cache row: the full-width cos/sin rows of `position` interleaved as
/// (cos_0, sin_0, cos_1, sin_1, ...), 2 * rotary values.
std::vector<float> rope_state_row(const std::vector<float>& table, size_t rotary, int32_t position) {
    const auto [cos, sin] = rope_rows(table, rotary, {position});
    std::vector<float> row(2 * rotary);
    for (size_t j = 0; j < rotary; j++) {
        row[2 * j] = cos[j];
        row[2 * j + 1] = sin[j];
    }
    return row;
}

/// `index_block_rope_cache` content: each block slot holds the interleaved row of its first
/// token, i.e. rope_state_row() of the slot's recorded position.
std::vector<float> rope_cache_rows(const std::vector<float>& table, size_t rotary, const std::vector<int32_t>& slot_positions) {
    std::vector<float> rows;
    rows.reserve(slot_positions.size() * 2 * rotary);
    for (int32_t position : slot_positions) {
        const auto row = rope_state_row(table, rotary, position);
        rows.insert(rows.end(), row.begin(), row.end());
    }
    return rows;
}

/// CPU reference for the split Indexer + Attention pair (H_ik == 1, num_selection_heads == 1,
/// query_block_size == 1). `positions[p]` is the RoPE position of absolute token p: query
/// RoPE uses its own position, a block summary uses the position of the block's first
/// token, whose row is what index_block_rope_cache records.
struct split_reference {
    split_dims d;
    size_t past_len = 0;
    size_t q_len = 0;

    std::vector<int32_t> positions;         // [total_len]
    std::vector<float> raw_index_keys_all;  // [total_len, Di]
    std::vector<float> index_query_cur;     // [q_len, Hiq, Di]
    std::vector<float> key_all;             // [total_len, Hkv*Dh]
    std::vector<float> value_all;           // [total_len, Hkv*Dv]
    std::vector<float> query_cur;           // [q_len, Hq*Dh]
    std::vector<float> q_norm, k_norm;
    std::vector<float> rotary;  // [max_positions, rotary], interleaved (cos_j, sin_j)

    mutable std::map<size_t, std::vector<float>> summary_memo;

    size_t total_len() const {
        return past_len + q_len;
    }

    std::vector<float> summary(size_t block_id) const {
        const auto it = summary_memo.find(block_id);
        if (it != summary_memo.end()) {
            return it->second;
        }
        std::vector<float> pooled(d.Di, 0.0f);
        for (size_t s = 0; s < d.r; s++) {
            const size_t p = block_id * d.r + s;
            for (size_t k = 0; k < d.Di; k++) {
                pooled[k] += raw_index_keys_all[p * d.Di + k];
            }
        }
        for (float& x : pooled) {
            // The pooled mean is rounded to T before RMSNorm (qsa-indexer.rst numerics).
            x = static_cast<float>(ov::float16(x / static_cast<float>(d.r)));
        }
        rms_norm(pooled, k_norm, d.eps);
        apply_rope(pooled, rotary, d.rotary, static_cast<size_t>(positions[block_id * d.r]));
        summary_memo.emplace(block_id, pooled);
        return pooled;
    }

    /// (score, block) pairs of every complete block visible to query `t`, best first.
    std::vector<std::pair<float, size_t>> scored_blocks(size_t t) const {
        const size_t abs_pos = past_len + t;
        const size_t n_complete = (abs_pos + 1) / d.r;

        std::vector<std::vector<float>> heads(d.Hiq);
        for (size_t h = 0; h < d.Hiq; h++) {
            heads[h].assign(index_query_cur.begin() + (t * d.Hiq + h) * d.Di, index_query_cur.begin() + (t * d.Hiq + h + 1) * d.Di);
            rms_norm(heads[h], q_norm, d.eps);
            apply_rope(heads[h], rotary, d.rotary, static_cast<size_t>(positions[abs_pos]));
        }

        std::vector<std::pair<float, size_t>> scored;
        scored.reserve(n_complete);
        for (size_t b = 0; b < n_complete; b++) {
            const auto k = summary(b);
            float total = 0.0f;
            for (size_t h = 0; h < d.Hiq; h++) {
                double dot = 0.0;
                for (size_t j = 0; j < d.Di; j++) {
                    dot += static_cast<double>(heads[h][j]) * k[j];
                }
                total += std::max(0.0f, static_cast<float>(dot));
            }
            scored.emplace_back(total / std::sqrt(static_cast<float>(d.Di)), b);
        }
        std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) {
                return a.first > b.first;
            }
            return a.second < b.second;
        });
        return scored;
    }

    /// Selected-block attention plus the causal tail, token-major [q_len, Hq*Dv].
    std::vector<float> attention(const std::vector<std::vector<size_t>>& selections) const {
        std::vector<float> out(q_len * d.Hq * d.Dv, 0.0f);
        const size_t group = d.Hq / d.Hkv;
        for (size_t t = 0; t < q_len; t++) {
            const size_t abs_pos = past_len + t;
            std::vector<size_t> tokens;
            for (size_t b : selections[t]) {
                for (size_t s = 0; s < d.r; s++) {
                    tokens.push_back(b * d.r + s);
                }
            }
            for (size_t p = ((abs_pos + 1) / d.r) * d.r; p <= abs_pos; p++) {
                tokens.push_back(p);
            }

            for (size_t h = 0; h < d.Hq; h++) {
                const size_t kv_head = h / group;
                const size_t q_base = t * d.Hq * d.Dh + h * d.Dh;

                std::vector<float> logits;
                logits.reserve(tokens.size());
                for (size_t p : tokens) {
                    double dot = 0.0;
                    for (size_t j = 0; j < d.Dh; j++) {
                        dot += static_cast<double>(query_cur[q_base + j]) * key_all[p * d.Hkv * d.Dh + kv_head * d.Dh + j];
                    }
                    logits.push_back(static_cast<float>(dot) / std::sqrt(static_cast<float>(d.Dh)));
                }
                const float m = *std::max_element(logits.begin(), logits.end());
                double sum = 0.0;
                for (float& x : logits) {
                    x = std::exp(x - m);
                    sum += x;
                }
                for (size_t j = 0; j < d.Dv; j++) {
                    double acc = 0.0;
                    for (size_t i = 0; i < tokens.size(); i++) {
                        acc += static_cast<double>(logits[i]) * value_all[tokens[i] * d.Hkv * d.Dv + kv_head * d.Dv + j];
                    }
                    out[(t * d.Hq + h) * d.Dv + j] = static_cast<float>(acc / sum);
                }
            }
        }
        return out;
    }
};

enum class Positions { Sequential, MRoPELike };

std::vector<int32_t> make_positions(size_t total, Positions kind) {
    std::vector<int32_t> pos(total);
    for (size_t p = 0; p < total; p++) {
        // Non-monotonic with duplicates, as an image span under mRoPE would produce.
        pos[p] = kind == Positions::Sequential ? static_cast<int32_t>(p) : static_cast<int32_t>(p / 2 + (p % 3 == 0 ? 3 : 0));
    }
    return pos;
}

split_reference make_split_reference(tests::random_generator& rg, const split_dims& dims, size_t past_len, size_t q_len, Positions kind, float scale = 1.0f) {
    split_reference ref;
    ref.d = dims;
    ref.past_len = past_len;
    ref.q_len = q_len;
    const size_t total = past_len + q_len;

    ref.positions = make_positions(total, kind);
    ref.d.max_positions = static_cast<size_t>(*std::max_element(ref.positions.begin(), ref.positions.end())) + 1;
    const auto& d = ref.d;

    ref.raw_index_keys_all = random_vec(rg, total * d.Di, scale);
    ref.index_query_cur = random_vec(rg, q_len * d.Hiq * d.Di, scale);
    ref.key_all = random_vec(rg, total * d.Hkv * d.Dh, scale);
    ref.value_all = random_vec(rg, total * d.Hkv * d.Dv, scale);
    ref.query_cur = random_vec(rg, q_len * d.Hq * d.Dh, scale);
    ref.q_norm = random_vec(rg, d.Di, 0.5f, 1.0f);
    ref.k_norm = random_vec(rg, d.Di, 0.5f, 1.0f);
    ref.rotary = make_rotary_table(d.max_positions, d.rotary);
    // Quantize every GPU-bound buffer through the same f16 round-trip `make_f16` applies,
    // so the reference and the GPU start from identical values (see quantize_f16's doc).
    for (auto* v : {&ref.raw_index_keys_all, &ref.index_query_cur, &ref.key_all, &ref.value_all, &ref.query_cur, &ref.q_norm, &ref.k_norm, &ref.rotary}) {
        quantize_f16(*v);
    }
    return ref;
}

/// Flat selection plan shared by both address modes: sel [q_len, K], count [q_len].
struct selection_plan {
    std::vector<int32_t> sel;
    std::vector<int32_t> count;
};

/// Synthetic plan for perf runs: `min(K, n_complete)` evenly spread complete blocks.
selection_plan synthetic_plan(const split_dims& d, size_t past_len, size_t q_len) {
    selection_plan plan;
    plan.sel.assign(q_len * d.block_topk, 0);
    plan.count.assign(q_len, 0);
    for (size_t t = 0; t < q_len; t++) {
        const size_t n_complete = (past_len + t + 1) / d.r;
        const size_t count = std::min(d.block_topk, n_complete);
        for (size_t i = 0; i < count; i++) {
            plan.sel[t * d.block_topk + i] = static_cast<int32_t>(i * n_complete / count);
        }
        plan.count[t] = static_cast<int32_t>(count);
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Paged pair (PagedQSAIndexer + SparsePA)
// ---------------------------------------------------------------------------

struct paged_inputs {
    split_dims d;
    size_t past_len = 0;
    size_t q_len = 0;
    size_t num_pages = 0;
    size_t spp = 0;  // summaries per page

    std::vector<float> index_query, raw_index_keys, query, key, value, rotary, q_norm, k_norm;
    std::vector<float> key_cache, value_cache, raw_k_cache, block_index_keys_cache;
    std::vector<int32_t> position_ids, position_cache, block_indices, block_indices_begins, subsequence_begins, past_lens, qb_start;
    selection_plan plan;  // only used when the attention runs without the indexer
};

void init_paged_metadata(paged_inputs& in, const split_dims& d, size_t past_len, size_t q_len) {
    in.d = d;
    in.past_len = past_len;
    in.q_len = q_len;
    const size_t total = past_len + q_len;
    in.num_pages = (total + d.page_size - 1) / d.page_size;
    in.spp = d.page_size / d.r;
    in.block_indices.resize(in.num_pages);
    std::iota(in.block_indices.begin(), in.block_indices.end(), 0);
    in.block_indices_begins = {0, static_cast<int32_t>(in.num_pages)};
    in.past_lens = {static_cast<int32_t>(past_len)};
    in.subsequence_begins = {0, static_cast<int32_t>(q_len)};
    in.qb_start = {0, static_cast<int32_t>(q_len)};
}

paged_inputs paged_inputs_from_reference(const split_reference& ref) {
    const auto& d = ref.d;
    paged_inputs in;
    init_paged_metadata(in, d, ref.past_len, ref.q_len);
    const size_t past_len = ref.past_len;

    in.key_cache.assign(in.num_pages * d.Hkv * d.page_size * d.Dh, 0.0f);
    in.value_cache.assign(in.num_pages * d.Hkv * d.page_size * d.Dv, 0.0f);
    in.raw_k_cache.assign(in.num_pages * d.page_size * d.Di, 0.0f);
    in.block_index_keys_cache.assign(in.num_pages * in.spp * d.Di, 0.0f);
    in.position_cache.assign(in.num_pages * in.spp, 0);

    for (size_t p = 0; p < past_len; p++) {
        const size_t page = p / d.page_size;
        const size_t slot = p % d.page_size;
        for (size_t h = 0; h < d.Hkv; h++) {
            for (size_t j = 0; j < d.Dh; j++) {
                in.key_cache[((page * d.Hkv + h) * d.page_size + slot) * d.Dh + j] = ref.key_all[p * d.Hkv * d.Dh + h * d.Dh + j];
            }
            for (size_t j = 0; j < d.Dv; j++) {
                in.value_cache[((page * d.Hkv + h) * d.page_size + slot) * d.Dv + j] = ref.value_all[p * d.Hkv * d.Dv + h * d.Dv + j];
            }
        }
        for (size_t j = 0; j < d.Di; j++) {
            in.raw_k_cache[(page * d.page_size + slot) * d.Di + j] = ref.raw_index_keys_all[p * d.Di + j];
        }
    }
    // Every block whose first token is history has its position recorded, including a
    // partial block that this call completes.
    for (size_t b = 0; b * d.r < past_len; b++) {
        const size_t page = b / in.spp;
        const size_t slot = b % in.spp;
        in.position_cache[page * in.spp + slot] = ref.positions[b * d.r];
        if (b * d.r + d.r <= past_len) {
            const auto s = ref.summary(b);
            std::copy(s.begin(), s.end(), in.block_index_keys_cache.begin() + (page * in.spp + slot) * d.Di);
        }
    }

    in.position_ids.assign(ref.positions.begin() + past_len, ref.positions.end());
    in.index_query = ref.index_query_cur;
    in.raw_index_keys.assign(ref.raw_index_keys_all.begin() + past_len * d.Di, ref.raw_index_keys_all.end());
    in.query = ref.query_cur;
    in.key.assign(ref.key_all.begin() + past_len * d.Hkv * d.Dh, ref.key_all.end());
    in.value.assign(ref.value_all.begin() + past_len * d.Hkv * d.Dv, ref.value_all.end());
    in.rotary = ref.rotary;
    in.q_norm = ref.q_norm;
    in.k_norm = ref.k_norm;
    return in;
}

paged_inputs paged_inputs_synthetic(const split_dims& dims, size_t past_len, size_t q_len, size_t batch = 1) {
    paged_inputs in;
    split_dims d = dims;
    d.max_positions = past_len + q_len;
    init_paged_metadata(in, d, past_len, q_len);
    in.key_cache = lcg_vec(in.num_pages * d.Hkv * d.page_size * d.Dh, 1);
    in.value_cache = lcg_vec(in.num_pages * d.Hkv * d.page_size * d.Dv, 2);
    in.raw_k_cache = lcg_vec(in.num_pages * d.page_size * d.Di, 3);
    in.block_index_keys_cache = lcg_vec(in.num_pages * in.spp * d.Di, 4);
    in.position_cache.resize(in.num_pages * in.spp);
    for (size_t b = 0; b < in.position_cache.size(); b++) {
        in.position_cache[b] = static_cast<int32_t>(std::min(b * d.r, d.max_positions - 1));
    }
    in.position_ids.resize(q_len);
    std::iota(in.position_ids.begin(), in.position_ids.end(), static_cast<int32_t>(past_len));
    in.index_query = lcg_vec(q_len * d.Hiq * d.Di, 5);
    in.raw_index_keys = lcg_vec(q_len * d.Di, 6);
    in.query = lcg_vec(q_len * d.Hq * d.Dh, 7);
    in.key = lcg_vec(q_len * d.Hkv * d.Dh, 8);
    in.value = lcg_vec(q_len * d.Hkv * d.Dv, 9);
    in.rotary = make_rotary_table(d.max_positions, d.rotary);
    in.q_norm = lcg_vec(d.Di, 10, 0.5f);
    in.k_norm = lcg_vec(d.Di, 11, 0.5f);
    in.plan = synthetic_plan(d, past_len, q_len);
    // `batch` sequences, each with its own copy of the pages and a contiguous page run.
    const size_t pages = in.num_pages;
    auto tile = [batch](auto& v) {
        const auto one = v;
        v.reserve(one.size() * batch);
        for (size_t b = 1; b < batch; b++) {
            v.insert(v.end(), one.begin(), one.end());
        }
    };
    tile(in.key_cache);
    tile(in.value_cache);
    tile(in.raw_k_cache);
    tile(in.block_index_keys_cache);
    tile(in.index_query);
    tile(in.raw_index_keys);
    tile(in.query);
    tile(in.key);
    tile(in.value);
    tile(in.position_cache);
    tile(in.position_ids);
    tile(in.plan.sel);
    tile(in.plan.count);
    in.block_indices.resize(pages * batch);
    std::iota(in.block_indices.begin(), in.block_indices.end(), 0);
    in.block_indices_begins.assign(batch + 1, 0);
    in.subsequence_begins.assign(batch + 1, 0);
    in.past_lens.assign(batch, static_cast<int32_t>(past_len));
    for (size_t b = 0; b < batch; b++) {
        in.block_indices_begins[b + 1] = static_cast<int32_t>((b + 1) * pages);
        in.subsequence_begins[b + 1] = static_cast<int32_t>((b + 1) * q_len);
    }
    in.qb_start = in.subsequence_begins;
    in.num_pages = pages * batch;
    in.q_len = q_len * batch;
    return in;
}

enum class Stage { Pair, IndexerOnly, AttentionOnly };

ExecutionConfig make_config(engine& engine) {
    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::optimize_data(true));
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    return cfg;
}

/// Adds a per-call tensor: a constant in a static topology, a network input (with the given
/// dynamic shape) in a dynamic one, where set_*_inputs() feeds it on every call.
void add_call_tensor(topology& topo, bool dynamic, const primitive_id& id, const ov::PartialShape& dyn_shape, const memory::ptr& mem) {
    if (dynamic) {
        topo.add(input_layout(id, layout{dyn_shape, mem->get_layout().data_type, format::bfyx}));
    } else {
        topo.add(data(id, mem));
    }
}

void set_paged_inputs(engine& engine, network& net, const paged_inputs& in, Stage stage, bool dynamic) {
    const auto& d = in.d;
    const auto q = I(in.q_len);
    const auto np = I(in.num_pages);
    if (dynamic) {
        net.set_input_data("block_indices", make_i32(engine, {np}, in.block_indices));
        net.set_input_data("block_indices_begins", make_i32(engine, {I(in.block_indices_begins.size())}, in.block_indices_begins));
        net.set_input_data("subsequence_begins", make_i32(engine, {I(in.subsequence_begins.size())}, in.subsequence_begins));
        net.set_input_data("past_lens", make_i32(engine, {I(in.past_lens.size())}, in.past_lens));
    }
    if (stage != Stage::AttentionOnly) {
        if (dynamic) {
            net.set_input_data("q_norm", make_f16(engine, {I(d.Di)}, in.q_norm));
            net.set_input_data("k_norm", make_f16(engine, {I(d.Di)}, in.k_norm));
        }
        const auto [rope_cos, rope_sin] = rope_rows(in.rotary, d.rotary, in.position_ids);
        net.set_input_data("index_query", make_f16(engine, {q, I(d.Hiq * d.Di)}, in.index_query));
        net.set_input_data("raw_index_keys", make_f16(engine, {q, I(d.Di)}, in.raw_index_keys));
        net.set_input_data("rope_cos", make_f16(engine, {q, I(d.rotary)}, rope_cos));
        net.set_input_data("rope_sin", make_f16(engine, {q, I(d.rotary)}, rope_sin));
        net.set_input_data("block_index_keys_cache", make_f16(engine, {np, 1, I(in.spp), I(d.Di)}, in.block_index_keys_cache));
        net.set_input_data("raw_k_cache", make_f16(engine, {np, 1, I(d.page_size), I(d.Di)}, in.raw_k_cache));
        net.set_input_data("rope_cache", make_f16(engine, {np, I(in.spp), I(2 * d.rotary)}, rope_cache_rows(in.rotary, d.rotary, in.position_cache)));
    }
    if (stage != Stage::IndexerOnly) {
        net.set_input_data("query", make_f16(engine, {q, I(d.Hq * d.Dh)}, in.query));
        net.set_input_data("key", make_f16(engine, {q, I(d.Hkv * d.Dh)}, in.key));
        net.set_input_data("value", make_f16(engine, {q, I(d.Hkv * d.Dv)}, in.value));
        net.set_input_data("key_cache", make_f16(engine, {np, I(d.Hkv), I(d.page_size), I(d.Dh)}, in.key_cache));
        net.set_input_data("value_cache", make_f16(engine, {np, I(d.Hkv), I(d.page_size), I(d.Dv)}, in.value_cache));
    }
}

/// `dynamic`: token/page/sequence extents are dynamic and every per-call tensor is an input,
/// so ONE network serves any (past_len, q_len) through the shape-agnostic impls.
std::unique_ptr<network> build_paged_network(engine& engine, const paged_inputs& in, Stage stage, bool dynamic = false) {
    const auto& d = in.d;
    const auto dyn = [dynamic](int64_t v) {
        return dynamic ? int64_t{-1} : v;
    };
    const auto q = dyn(I(in.q_len));
    const auto np = dyn(I(in.num_pages));
    topology topo;
    add_call_tensor(topo, dynamic, "block_indices", {-1}, make_i32(engine, {I(in.num_pages)}, in.block_indices));
    add_call_tensor(topo, dynamic, "block_indices_begins", {-1}, make_i32(engine, {I(in.block_indices_begins.size())}, in.block_indices_begins));
    add_call_tensor(topo, dynamic, "subsequence_begins", {-1}, make_i32(engine, {I(in.subsequence_begins.size())}, in.subsequence_begins));
    add_call_tensor(topo, dynamic, "past_lens", {-1}, make_i32(engine, {I(in.past_lens.size())}, in.past_lens));

    if (stage != Stage::AttentionOnly) {
        ov::intel_gpu::op::PagedQSAIndexer::Config cfg;
        cfg.key_block_size = d.r;
        cfg.block_topk = d.block_topk;
        cfg.num_selection_heads = 1;
        cfg.indexer_rotary_dim = d.rotary;
        cfg.indexer_head_dim = d.Di;
        cfg.epsilon = d.eps;
        cfg.scale = 1.0f / std::sqrt(static_cast<float>(d.Di));

        topo.add(input_layout("index_query", layout{{q, I(d.Hiq * d.Di)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("raw_index_keys", layout{{q, I(d.Di)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("rope_cos", layout{{q, I(d.rotary)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("rope_sin", layout{{q, I(d.rotary)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("block_index_keys_cache", layout{{np, 1, I(in.spp), I(d.Di)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("raw_k_cache", layout{{np, 1, I(d.page_size), I(d.Di)}, data_types::f16, format::bfyx}));
        add_call_tensor(topo, dynamic, "q_norm", {I(d.Di)}, make_f16(engine, {I(d.Di)}, in.q_norm));
        add_call_tensor(topo, dynamic, "k_norm", {I(d.Di)}, make_f16(engine, {I(d.Di)}, in.k_norm));
        topo.add(input_layout("rope_cache", layout{{np, I(in.spp), I(2 * d.rotary)}, data_types::f16, format::bfyx}));
        static_assert(PagedIn::COUNT == 13, "keep the test input order in sync with PagedQSAIndexer::InputIdx");
        topo.add(paged_qsa_indexer("indexer",
                                   {input_info("index_query"),
                                    input_info("raw_index_keys"),
                                    input_info("rope_cos"),
                                    input_info("rope_sin"),
                                    input_info("q_norm"),
                                    input_info("k_norm"),
                                    input_info("past_lens"),
                                    input_info("subsequence_begins"),
                                    input_info("block_indices"),
                                    input_info("block_indices_begins"),
                                    input_info("raw_k_cache"),
                                    input_info("rope_cache"),
                                    input_info("block_index_keys_cache")},
                                   cfg));
    }

    if (stage != Stage::IndexerOnly) {
        cldnn::sparse_pa::Config cfg;
        cfg.num_heads = d.Hq;
        cfg.num_kv_heads = d.Hkv;
        cfg.num_selection_heads = 1;
        cfg.k_head_size = d.Dh;
        cfg.v_head_size = d.Dv;
        cfg.key_block_size = d.r;
        cfg.query_block_size = 1;
        cfg.causal = true;
        const float scale = 1.0f / std::sqrt(static_cast<float>(d.Dh));

        topo.add(input_layout("query", layout{{q, I(d.Hq * d.Dh)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("key", layout{{q, I(d.Hkv * d.Dh)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("value", layout{{q, I(d.Hkv * d.Dv)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("key_cache", layout{{np, I(d.Hkv), I(d.page_size), I(d.Dh)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("value_cache", layout{{np, I(d.Hkv), I(d.page_size), I(d.Dv)}, data_types::f16, format::bfyx}));
        input_info sel("indexer", 0), count("indexer", 1);
        if (stage == Stage::AttentionOnly) {
            topo.add(data("sel", make_i32(engine, {I(in.q_len), 1, I(d.block_topk)}, in.plan.sel)));
            topo.add(data("count", make_i32(engine, {I(in.q_len), 1}, in.plan.count)));
            sel = input_info("sel");
            count = input_info("count");
        }
        // The primitive takes QUERY..SELECTED_BLOCK_COUNT only: scale is folded in, sinks are unsupported, qb_start is unused.
        static_assert(SparsePAIn::SELECTED_BLOCK_COUNT == 10 && SparsePAIn::PAST_LENS == 5, "keep the test input order in sync with PagedSparseAttention");
        topo.add(sparse_pa("attn",
                           {input_info("query"),
                            input_info("key"),
                            input_info("value"),
                            input_info("key_cache"),
                            input_info("value_cache"),
                            input_info("past_lens"),
                            input_info("subsequence_begins"),
                            input_info("block_indices"),
                            input_info("block_indices_begins"),
                            sel,
                            count},
                           cfg,
                           scale));
    }

    auto net = std::make_unique<network>(engine, topo, make_config(engine));
    set_paged_inputs(engine, *net, in, stage, dynamic);
    return net;
}

// ---------------------------------------------------------------------------
// Dense pair (QSAIndexer + SparseSDPA). Every batch row has its own reference; a row
// shorter than the shared query length L is a fresh sequence, left-padded to L, and its
// K/V are left-padded to the longest row's length S (SparseSDPA's key_length contract).
// ---------------------------------------------------------------------------

template <typename T>
void append(std::vector<T>& dst, const std::vector<T>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

/// Storage the ops must never read (left padding, unused state slots): any read poisons
/// the output with NaN.
const float poison = std::numeric_limits<float>::quiet_NaN();

struct dense_inputs {
    split_dims d;
    size_t batch = 0;
    size_t q_len = 0;        // L, shared by every row
    size_t kv_len = 0;       // S, the longest row; shorter rows are left-padded
    size_t summary_cap = 1;  // floor(S_cap / r) summary entries, uniform across rows

    std::vector<int32_t> key_length;  // [B], valid tokens after the call
    std::vector<float> index_query, raw_index_keys, rope_cos, rope_sin, q_norm, k_norm;
    std::vector<float> block_index_keys_cache;  // [B, 1, summary_cap, Di]
    std::vector<float> raw_state;               // [B, 1, r - 1, Di], slot p % r of the incomplete block
    std::vector<float> rope_state;              // [B, 2 * rotary], first-token row of the incomplete block
    std::vector<float> query_bhld, key_bhsd, value_bhsd;
    selection_plan plan;  // only used when the attention runs without the indexer
};

/// Left-padding rows of batch row `b`.
size_t dense_pad(const dense_inputs& in, size_t b) {
    return in.q_len - std::min(in.q_len, static_cast<size_t>(in.key_length[b]));
}

/// Appends batch row `ref`: inputs left-padded to `in.q_len`, K/V left-padded to `in.kv_len`,
/// and the state a previous call would have returned after `ref.past_len` tokens.
void append_dense_row(dense_inputs& in, const split_reference& ref) {
    const auto& d = in.d;
    const size_t L = in.q_len;
    const size_t S = in.kv_len;
    const size_t past = ref.past_len;
    const size_t total = ref.total_len();
    ASSERT_TRUE(ref.q_len <= L && total <= S && (ref.q_len == L || past == 0)) << "only a new sequence can be left-padded";
    const size_t pad = L - ref.q_len;
    const size_t pad_k = S - total;
    in.key_length.push_back(static_cast<int32_t>(total));

    auto left_padded = [pad](const std::vector<float>& rows, size_t width) {
        std::vector<float> out(pad * width, poison);
        append(out, rows);
        return out;
    };
    append(in.index_query, left_padded(ref.index_query_cur, d.Hiq * d.Di));
    append(in.raw_index_keys, left_padded(std::vector<float>(ref.raw_index_keys_all.begin() + past * d.Di, ref.raw_index_keys_all.end()), d.Di));
    const auto [cos, sin] = rope_rows(ref.rotary, d.rotary, std::vector<int32_t>(ref.positions.begin() + past, ref.positions.end()));
    append(in.rope_cos, left_padded(cos, d.rotary));
    append(in.rope_sin, left_padded(sin, d.rotary));

    std::vector<float> summary(in.summary_cap * d.Di, 0.0f);
    for (size_t b = 0; b < past / d.r; b++) {
        const auto s = ref.summary(b);
        std::copy(s.begin(), s.end(), summary.begin() + b * d.Di);
    }
    append(in.block_index_keys_cache, summary);
    const size_t staged = past % d.r;
    const size_t first = past - staged;
    std::vector<float> raw((d.r - 1) * d.Di, poison);
    std::copy(ref.raw_index_keys_all.begin() + first * d.Di, ref.raw_index_keys_all.begin() + past * d.Di, raw.begin());
    append(in.raw_state, raw);
    append(in.rope_state, staged ? rope_state_row(ref.rotary, d.rotary, ref.positions[first]) : std::vector<float>(2 * d.rotary, poison));

    std::vector<float> query(d.Hq * L * d.Dh, poison);
    for (size_t t = 0; t < ref.q_len; t++) {
        for (size_t h = 0; h < d.Hq; h++) {
            for (size_t j = 0; j < d.Dh; j++) {
                query[(h * L + pad + t) * d.Dh + j] = ref.query_cur[(t * d.Hq + h) * d.Dh + j];
            }
        }
    }
    append(in.query_bhld, query);
    std::vector<float> key(d.Hkv * S * d.Dh, poison);
    std::vector<float> value(d.Hkv * S * d.Dv, poison);
    for (size_t p = 0; p < total; p++) {
        for (size_t h = 0; h < d.Hkv; h++) {
            for (size_t j = 0; j < d.Dh; j++) {
                key[(h * S + pad_k + p) * d.Dh + j] = ref.key_all[p * d.Hkv * d.Dh + h * d.Dh + j];
            }
            for (size_t j = 0; j < d.Dv; j++) {
                value[(h * S + pad_k + p) * d.Dv + j] = ref.value_all[p * d.Hkv * d.Dv + h * d.Dv + j];
            }
        }
    }
    append(in.key_bhsd, key);
    append(in.value_bhsd, value);
}

dense_inputs dense_inputs_from_references(const std::vector<split_reference>& refs) {
    dense_inputs in;
    in.d = refs.front().d;
    in.batch = refs.size();
    size_t blocks = 0;
    for (const auto& ref : refs) {
        in.q_len = std::max(in.q_len, ref.q_len);
        in.kv_len = std::max(in.kv_len, ref.total_len());
        blocks = std::max(blocks, ref.total_len() / in.d.r);
    }
    in.summary_cap = std::max<size_t>(1, blocks);
    in.q_norm = refs.front().q_norm;
    in.k_norm = refs.front().k_norm;
    for (const auto& ref : refs) {
        append_dense_row(in, ref);
    }
    return in;
}

dense_inputs dense_inputs_from_reference(const split_reference& ref) {
    return dense_inputs_from_references({ref});
}

dense_inputs dense_inputs_synthetic(const split_dims& dims, size_t past_len, size_t q_len, size_t batch = 1) {
    dense_inputs in;
    in.d = dims;
    const auto& d = in.d;
    const size_t total = past_len + q_len;
    in.batch = 1;
    in.q_len = q_len;
    in.kv_len = total;
    in.summary_cap = std::max<size_t>(1, total / d.r);
    in.key_length = {static_cast<int32_t>(total)};
    const auto rotary = make_rotary_table(total, d.rotary);
    std::vector<int32_t> positions(q_len);
    std::iota(positions.begin(), positions.end(), static_cast<int32_t>(past_len));
    std::tie(in.rope_cos, in.rope_sin) = rope_rows(rotary, d.rotary, positions);
    in.block_index_keys_cache = lcg_vec(in.summary_cap * d.Di, 4);
    in.raw_state = lcg_vec((d.r - 1) * d.Di, 3);
    in.rope_state = rope_state_row(rotary, d.rotary, static_cast<int32_t>(past_len - past_len % d.r));
    in.index_query = lcg_vec(q_len * d.Hiq * d.Di, 5);
    in.raw_index_keys = lcg_vec(q_len * d.Di, 6);
    in.query_bhld = lcg_vec(d.Hq * q_len * d.Dh, 7);
    in.key_bhsd = lcg_vec(d.Hkv * total * d.Dh, 8);
    in.value_bhsd = lcg_vec(d.Hkv * total * d.Dv, 9);
    in.q_norm = lcg_vec(d.Di, 10, 0.5f);
    in.k_norm = lcg_vec(d.Di, 11, 0.5f);
    in.plan = synthetic_plan(d, past_len, q_len);
    // Every batch row repeats row 0's content (all tensors are batch-major).
    in.batch = batch;
    for (auto* v : {&in.index_query,
                    &in.raw_index_keys,
                    &in.rope_cos,
                    &in.rope_sin,
                    &in.block_index_keys_cache,
                    &in.raw_state,
                    &in.rope_state,
                    &in.query_bhld,
                    &in.key_bhsd,
                    &in.value_bhsd}) {
        const auto row = *v;
        v->reserve(row.size() * batch);
        for (size_t b = 1; b < batch; b++) {
            append(*v, row);
        }
    }
    in.key_length.assign(batch, static_cast<int32_t>(total));
    return in;
}

void set_dense_inputs(engine& engine, network& net, const dense_inputs& in, Stage stage, bool dynamic) {
    const auto& d = in.d;
    const auto B = I(in.batch);
    const auto q = I(in.q_len);
    const auto S = I(in.kv_len);
    if (dynamic) {
        net.set_input_data("key_length", make_i32(engine, {B}, in.key_length));
    }
    if (stage != Stage::AttentionOnly) {
        if (dynamic) {
            net.set_input_data("q_norm", make_f16(engine, {I(d.Di)}, in.q_norm));
            net.set_input_data("k_norm", make_f16(engine, {I(d.Di)}, in.k_norm));
        }
        net.set_input_data("index_query", make_f16(engine, {B, q, I(d.Hiq), I(d.Di)}, in.index_query));
        net.set_input_data("raw_index_keys", make_f16(engine, {B, q, 1, I(d.Di)}, in.raw_index_keys));
        net.set_input_data("rope_cos", make_f16(engine, {B, q, I(d.rotary)}, in.rope_cos));
        net.set_input_data("rope_sin", make_f16(engine, {B, q, I(d.rotary)}, in.rope_sin));
        net.set_input_data("block_index_keys_cache", make_f16(engine, {B, 1, I(in.summary_cap), I(d.Di)}, in.block_index_keys_cache));
        net.set_input_data("raw_state", make_f16(engine, {B, 1, I(d.r - 1), I(d.Di)}, in.raw_state));
        net.set_input_data("rope_state", make_f16(engine, {B, I(2 * d.rotary)}, in.rope_state));
    }
    if (stage != Stage::IndexerOnly) {
        net.set_input_data("query", make_f16(engine, {B, I(d.Hq), q, I(d.Dh)}, in.query_bhld));
        net.set_input_data("key", make_f16(engine, {B, I(d.Hkv), S, I(d.Dh)}, in.key_bhsd));
        net.set_input_data("value", make_f16(engine, {B, I(d.Hkv), S, I(d.Dv)}, in.value_bhsd));
    }
}

/// `dynamic`: B/L/S and the summary capacity are dynamic and every per-call tensor is an
/// input, so ONE network serves any (batch, key_length, q_len) through the shape-agnostic impls.
std::unique_ptr<network> build_dense_network(engine& engine, const dense_inputs& in, Stage stage, bool dynamic = false) {
    const auto& d = in.d;
    const auto dyn = [dynamic](int64_t v) {
        return dynamic ? int64_t{-1} : v;
    };
    const auto B = dyn(I(in.batch));
    const auto q = dyn(I(in.q_len));
    const auto S = dyn(I(in.kv_len));
    const auto summary_cap = dyn(I(in.summary_cap));
    topology topo;
    add_call_tensor(topo, dynamic, "key_length", {-1}, make_i32(engine, {I(in.batch)}, in.key_length));

    if (stage != Stage::AttentionOnly) {
        ov::intel_gpu::op::QSAIndexer::Config cfg;
        cfg.key_block_size = d.r;
        cfg.block_topk = d.block_topk;
        cfg.num_selection_heads = 1;
        cfg.indexer_rotary_dim = d.rotary;
        cfg.indexer_head_dim = d.Di;
        cfg.epsilon = d.eps;
        cfg.scale = 1.0f / std::sqrt(static_cast<float>(d.Di));

        topo.add(input_layout("index_query", layout{{B, q, I(d.Hiq), I(d.Di)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("raw_index_keys", layout{{B, q, 1, I(d.Di)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("rope_cos", layout{{B, q, I(d.rotary)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("rope_sin", layout{{B, q, I(d.rotary)}, data_types::f16, format::bfyx}));
        add_call_tensor(topo, dynamic, "q_norm", {I(d.Di)}, make_f16(engine, {I(d.Di)}, in.q_norm));
        add_call_tensor(topo, dynamic, "k_norm", {I(d.Di)}, make_f16(engine, {I(d.Di)}, in.k_norm));
        topo.add(input_layout("block_index_keys_cache", layout{{B, 1, summary_cap, I(d.Di)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("raw_state", layout{{B, 1, I(d.r - 1), I(d.Di)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("rope_state", layout{{B, I(2 * d.rotary)}, data_types::f16, format::bfyx}));
        static_assert(DenseIdxIn::COUNT == 10, "keep the test input order in sync with QSAIndexer::InputIdx");
        topo.add(qsa_indexer("indexer",
                             {input_info("index_query"),
                              input_info("raw_index_keys"),
                              input_info("rope_cos"),
                              input_info("rope_sin"),
                              input_info("key_length"),
                              input_info("q_norm"),
                              input_info("k_norm"),
                              input_info("block_index_keys_cache"),
                              input_info("raw_state"),
                              input_info("rope_state")},
                             cfg));
    }

    if (stage != Stage::IndexerOnly) {
        ov::intel_gpu::op::SparseSDPA::Config cfg;
        cfg.num_heads = d.Hq;
        cfg.num_kv_heads = d.Hkv;
        cfg.num_selection_heads = 1;
        cfg.k_head_size = d.Dh;
        cfg.v_head_size = d.Dv;
        cfg.key_block_size = d.r;
        cfg.query_block_size = 1;
        cfg.causal = true;

        topo.add(input_layout("query", layout{{B, I(d.Hq), q, I(d.Dh)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("key", layout{{B, I(d.Hkv), S, I(d.Dh)}, data_types::f16, format::bfyx}));
        topo.add(input_layout("value", layout{{B, I(d.Hkv), S, I(d.Dv)}, data_types::f16, format::bfyx}));
        input_info sel("indexer", 0), count("indexer", 1);
        if (stage == Stage::AttentionOnly) {
            topo.add(data("sel", make_i32(engine, {1, 1, I(in.q_len), I(d.block_topk)}, in.plan.sel)));
            topo.add(data("count", make_i32(engine, {1, 1, I(in.q_len)}, in.plan.count)));
            sel = input_info("sel");
            count = input_info("count");
        }
        static_assert(SparseSDPAIn::COUNT_MIN == 6, "keep the test input order in sync with SparseSDPA::InputIdx");
        topo.add(sparse_sdpa("attn",
                             {input_info("query"), input_info("key"), input_info("value"), sel, count, input_info("key_length")},
                             cfg,
                             1.0f / std::sqrt(static_cast<float>(d.Dh))));
    }

    auto net = std::make_unique<network>(engine, topo, make_config(engine));
    set_dense_inputs(engine, *net, in, stage, dynamic);
    return net;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

struct pair_outputs {
    selection_plan plan;      // GPU indexer outputs, flat
    std::vector<float> attn;  // token-major [q_len, Hq*Dv]
    std::vector<std::string> indexer_kernels, attention_kernels;
};

/// Kernels the primitive's last execute() enqueued, in order, without the `_ocl_<hash>` suffix.
std::vector<std::string> executed_kernels(network& net, const primitive_id& id) {
    auto inst = net.get_primitive(id);
    std::istringstream entries(inst->get_impl()->get_kernels_dump_info(*inst->get_impl_params()).get_entries());
    std::vector<std::string> names;
    for (std::string entry; entries >> entry;) {
        names.push_back(entry.substr(0, entry.rfind("_ocl_")));
    }
    return names;
}

void read_pair_outputs(network& net, pair_outputs& out) {
    auto indexer = net.get_primitive("indexer");
    out.plan.sel = read_i32(indexer->output_memory_ptr(0));
    out.plan.count = read_i32(indexer->output_memory_ptr(1));
    // Dynamic networks may keep a larger buffer from an earlier call.
    out.plan.sel.resize(indexer->get_output_layout(0).count());
    out.plan.count.resize(indexer->get_output_layout(1).count());
    out.indexer_kernels = executed_kernels(net, "indexer");
    out.attention_kernels = executed_kernels(net, "attn");
}

enum class IndexerPath { Decode, Dpas, Fused };
enum class AttnPath { Scalar, Dpas, GqaDpas, GqaSplit, Split };

/// Expected kernel routing of one pair execution; `decode_chunks` counts BLOCK_SCORES row chunks.
struct route {
    IndexerPath indexer;
    AttnPath attention;
    size_t decode_chunks = 1;
};

std::vector<std::string> expected_indexer_kernels(bool paged, const route& r) {
    std::vector<std::string> k = {paged ? "qsa_indexer_prepare_paged" : "qsa_indexer_prepare_dense"};
    switch (r.indexer) {
    case IndexerPath::Decode:
        for (size_t i = 0; i < r.decode_chunks; i++) {
            k.insert(k.end(), {"qsa_indexer_score_partition", "qsa_indexer_topk_decode"});
        }
        break;
    case IndexerPath::Dpas:
        k.insert(k.end(), {"qsa_indexer_score_dpas", "qsa_indexer_topk"});
        break;
    case IndexerPath::Fused:
        // There is no single-thread fused kernel: partition scorer + prefill top-k.
        k.insert(k.end(), {"qsa_indexer_score_partition", "qsa_indexer_topk"});
        break;
    }
    return k;
}

std::vector<std::string> expected_attention_kernels(bool paged, const route& r) {
    const std::string base = paged ? "sparse_attention_paged" : "sparse_attention_dense";
    std::vector<std::string> k;
    if (paged) {
        k.push_back("sparse_attention_kv_write");
    }
    switch (r.attention) {
    case AttnPath::Scalar:
        k.push_back(base);
        break;
    case AttnPath::Dpas:
        k.push_back(base + "_dpas");
        break;
    case AttnPath::GqaDpas:
        k.push_back(base + "_gqa_dpas");
        break;
    case AttnPath::GqaSplit:
        k.insert(k.end(), {base + "_gqa_dpas_split", base + "_split_reduce"});
        break;
    case AttnPath::Split:
        k.insert(k.end(), {base + "_split", base + "_split_reduce"});
        break;
    }
    return k;
}

/// Mirrors qsa::xmx_allowed(): the systolic kernels are only routed on Xe2+ (so not on DG2) and can be switched off.
bool xmx_routes_enabled() {
    const char* v = std::getenv("OV_GPU_QSA_OCL_DISABLE_XMX");
    if (v != nullptr && std::string(v) == "1") {
        return false;
    }
    const auto& info = get_test_engine().get_device_info();
    return info.arch >= gpu_arch::xe2 && info.supports_immad;
}

/// Route the plugin takes without XMX; `kv_len` is the dense key length (SparseSDPA splits decode from 2048 on).
route resolve_route(const route& r, bool paged, size_t kv_len) {
    if (xmx_routes_enabled()) {
        return r;
    }
    constexpr size_t dense_split_min_kv = 2048;
    route out = r;
    if (out.indexer == IndexerPath::Dpas) {
        out.indexer = IndexerPath::Fused;
    }
    if (out.attention == AttnPath::Dpas || out.attention == AttnPath::GqaDpas) {
        out.attention = AttnPath::Scalar;
    } else if (out.attention == AttnPath::GqaSplit) {
        out.attention = (paged || kv_len >= dense_split_min_kv) ? AttnPath::Split : AttnPath::Scalar;
    }
    return out;
}

void expect_route(bool paged, const pair_outputs& gpu, const route& expected, size_t kv_len = 0) {
    const auto r = resolve_route(expected, paged, kv_len);
    EXPECT_EQ(gpu.indexer_kernels, expected_indexer_kernels(paged, r));
    EXPECT_EQ(gpu.attention_kernels, expected_attention_kernels(paged, r));
}

pair_outputs run_paged_pair(engine& engine, const split_reference& ref) {
    auto net = build_paged_network(engine, paged_inputs_from_reference(ref), Stage::Pair);
    auto outputs = net->execute();
    pair_outputs out;
    out.attn = read_f16(outputs.at("attn").get_memory());
    read_pair_outputs(*net, out);
    return out;
}

/// [Hq, L, Dv] (B == 1) -> token-major [L, Hq*Dv], the reference's layout.
std::vector<float> bhld_to_token_major(const std::vector<float>& bhld, const split_dims& d, size_t q_len) {
    std::vector<float> out(bhld.size());
    for (size_t t = 0; t < q_len; t++) {
        for (size_t h = 0; h < d.Hq; h++) {
            for (size_t j = 0; j < d.Dv; j++) {
                out[(t * d.Hq + h) * d.Dv + j] = bhld[(h * q_len + t) * d.Dv + j];
            }
        }
    }
    return out;
}

/// Combined absolute+relative error summary (numpy-`allclose`-style), plus a non-finite
/// count so a corrupted/overflowing kernel output fails loudly instead of silently
/// skewing the mean.
struct error_stats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double mean_abs = 0.0;
    size_t count = 0;
    size_t non_finite = 0;
};

error_stats compute_error_stats(const std::vector<float>& actual, const std::vector<float>& expected) {
    error_stats s;
    double sum_abs = 0.0;
    for (size_t i = 0; i < expected.size(); i++) {
        if (!std::isfinite(actual[i])) {
            s.non_finite++;
            continue;
        }
        const double err = std::fabs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        sum_abs += err;
        s.max_abs = std::max(s.max_abs, err);
        s.max_rel = std::max(s.max_rel, err / std::max(1e-6, std::fabs(static_cast<double>(expected[i]))));
        s.count++;
    }
    s.mean_abs = s.count ? sum_abs / static_cast<double>(s.count) : 0.0;
    return s;
}

void print_error_stats(const char* label, const error_stats& s) {
    std::cout << "QSA_PRECISION " << label << " n=" << s.count << " non_finite=" << s.non_finite << " max_abs=" << s.max_abs << " mean_abs=" << s.mean_abs
              << " max_rel=" << s.max_rel << std::endl;
}

/// With QSA_DUMP_DIR set, writes the GPU attention output and selection plan of every validated pair so that two runs
/// can be diffed directly.
void dump_pair_outputs(const pair_outputs& gpu) {
    const char* dir = std::getenv("QSA_DUMP_DIR");
    if (dir == nullptr) {
        return;
    }
    static int counter = 0;
    const std::string base = std::string(dir) + "/" + ::testing::UnitTest::GetInstance()->current_test_info()->name() + "_" + std::to_string(counter++);
    auto write = [&](const std::string& suffix, const auto& v) {
        std::ofstream f(base + suffix, std::ios::binary);
        f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(v[0])));
    };
    write(".attn.f32", gpu.attn);
    write(".sel.i32", gpu.plan.sel);
    write(".count.i32", gpu.plan.count);
}

/// Checks the GPU plan against the reference scores (count exact; every selected block
/// complete, unique, and scoring within fp16 tolerance of the k-th best), then compares
/// attention computed by the reference over that same plan using a combined
/// absolute+relative tolerance (`atol + rtol * |ref|`), after asserting every output
/// element is finite.
void validate_pair(const split_reference& ref, const pair_outputs& gpu, float atol, float rtol) {
    dump_pair_outputs(gpu);
    const auto& d = ref.d;
    ASSERT_EQ(gpu.plan.count.size(), ref.q_len);
    ASSERT_EQ(gpu.plan.sel.size(), ref.q_len * d.block_topk);
    std::vector<std::vector<size_t>> selections(ref.q_len);
    for (size_t t = 0; t < ref.q_len; t++) {
        const auto scored = ref.scored_blocks(t);
        const size_t expected_count = std::min(d.block_topk, scored.size());
        ASSERT_EQ(static_cast<size_t>(gpu.plan.count[t]), expected_count) << "t=" << t;
        if (expected_count == 0) {
            continue;
        }
        std::vector<float> score_of(scored.size());
        for (const auto& [score, block] : scored) {
            score_of[block] = score;
        }
        const float kth = scored[expected_count - 1].first;
        const float tol = 1e-2f * std::max(1.0f, std::fabs(kth));
        std::set<size_t> seen;
        for (size_t i = 0; i < expected_count; i++) {
            const int32_t b = gpu.plan.sel[t * d.block_topk + i];
            ASSERT_GE(b, 0) << "t=" << t << " i=" << i;
            ASSERT_LT(static_cast<size_t>(b), scored.size()) << "t=" << t << " selected an incomplete block";
            ASSERT_TRUE(seen.insert(static_cast<size_t>(b)).second) << "t=" << t << " selected block " << b << " twice";
            EXPECT_GE(score_of[b], kth - tol) << "t=" << t << " block " << b << " is not in the top-" << expected_count;
        }
        selections[t].assign(seen.begin(), seen.end());
    }

    const auto expected = ref.attention(selections);
    ASSERT_EQ(gpu.attn.size(), expected.size());
    const auto stats = compute_error_stats(gpu.attn, expected);
    print_error_stats("attention", stats);
    EXPECT_EQ(stats.non_finite, 0u) << "attention output contains non-finite values";

    size_t mismatches = 0;
    for (size_t i = 0; i < expected.size(); i++) {
        const float tol = atol + rtol * std::fabs(expected[i]);
        if (std::isfinite(gpu.attn[i]) && std::fabs(gpu.attn[i] - expected[i]) > tol && mismatches++ < 8) {
            ADD_FAILURE() << "attn[" << i << "] (t=" << i / (d.Hq * d.Dv) << ") gpu=" << gpu.attn[i] << " ref=" << expected[i] << " tol=" << tol;
        }
    }
    EXPECT_EQ(mismatches, 0u);
}

/// Checks the dense state outputs against `refs`: the summary of every complete block, and
/// the staged raw keys and first-token RoPE row of the new incomplete block.
void validate_dense_state(network& net, const dense_inputs& in, const std::vector<split_reference>& refs) {
    using Out = ov::intel_gpu::op::QSAIndexer::OutputIdx;
    const auto& d = in.d;
    auto indexer = net.get_primitive("indexer");
    const auto summary = read_f16(indexer->output_memory_ptr(Out::BLOCK_INDEX_KEYS_CACHE));
    const auto raw = read_f16(indexer->output_memory_ptr(Out::PAST_INCOMPLETE_RAW_K));
    const auto rope = read_f16(indexer->output_memory_ptr(Out::PAST_INCOMPLETE_ROPE));
    ASSERT_EQ(indexer->get_output_layout(Out::BLOCK_INDEX_KEYS_CACHE).count(), in.batch * in.summary_cap * d.Di);
    ASSERT_EQ(indexer->get_output_layout(Out::PAST_INCOMPLETE_RAW_K).count(), in.batch * (d.r - 1) * d.Di);
    ASSERT_EQ(indexer->get_output_layout(Out::PAST_INCOMPLETE_ROPE).count(), in.batch * 2 * d.rotary);
    for (size_t b = 0; b < refs.size(); b++) {
        const auto& ref = refs[b];
        const size_t total = ref.total_len();
        SCOPED_TRACE("state of row " + std::to_string(b) + " key_length=" + std::to_string(total));
        size_t mismatches = 0;
        for (size_t blk = 0; blk < total / d.r; blk++) {
            const auto expected = ref.summary(blk);
            for (size_t j = 0; j < d.Di; j++) {
                const float got = summary[(b * in.summary_cap + blk) * d.Di + j];
                if (!(std::fabs(got - expected[j]) <= 1e-2f + 1e-2f * std::fabs(expected[j])) && mismatches++ < 8) {
                    ADD_FAILURE() << "summary[" << blk << "][" << j << "] gpu=" << got << " ref=" << expected[j];
                }
            }
        }
        const size_t staged = total % d.r;
        const size_t first = total - staged;
        for (size_t i = 0; i < staged * d.Di; i++) {
            if (raw[b * (d.r - 1) * d.Di + i] != ref.raw_index_keys_all[first * d.Di + i] && mismatches++ < 8) {
                ADD_FAILURE() << "staged key " << i / d.Di << "[" << i % d.Di << "] gpu=" << raw[b * (d.r - 1) * d.Di + i]
                              << " ref=" << ref.raw_index_keys_all[first * d.Di + i];
            }
        }
        if (staged != 0) {
            const auto expected = rope_state_row(ref.rotary, d.rotary, ref.positions[first]);
            for (size_t j = 0; j < expected.size(); j++) {
                if (rope[b * 2 * d.rotary + j] != static_cast<float>(ov::float16(expected[j])) && mismatches++ < 8) {
                    ADD_FAILURE() << "rope state[" << j << "] gpu=" << rope[b * 2 * d.rotary + j] << " ref=" << expected[j];
                }
            }
        }
        EXPECT_EQ(mismatches, 0u);
    }
}

/// Validates a dense network execution row by row: the real rows of each batch row against
/// its reference, the left-padding rows (no selection, zero attention), and the state.
void validate_dense_outputs(network& net, const dense_inputs& in, const std::vector<split_reference>& refs, const route* r, float atol, float rtol) {
    const auto& d = in.d;
    const size_t L = in.q_len;
    const size_t K = d.block_topk;
    // The test config may use an out-of-order queue: wait for every kernel before reading back.
    net.get_stream().finish();
    pair_outputs all;
    auto attn_inst = net.get_primitive("attn");
    auto attn = read_f16(attn_inst->output_memory_ptr(0));
    attn.resize(attn_inst->get_output_layout().count());
    read_pair_outputs(net, all);
    if (r != nullptr) {
        expect_route(false, all, *r, in.kv_len);
    }
    ASSERT_EQ(all.plan.count.size(), in.batch * L);
    ASSERT_EQ(attn.size(), in.batch * d.Hq * L * d.Dv);
    for (size_t b = 0; b < refs.size(); b++) {
        const auto& ref = refs[b];
        const size_t pad = dense_pad(in, b);
        SCOPED_TRACE("row " + std::to_string(b) + " past_len=" + std::to_string(ref.past_len) + " q_len=" + std::to_string(ref.q_len));
        size_t bad_padding = 0;
        for (size_t l = 0; l < pad; l++) {
            bad_padding += static_cast<size_t>(all.plan.count[b * L + l] != 0);
            for (size_t i = 0; i < K; i++) {
                bad_padding += static_cast<size_t>(all.plan.sel[(b * L + l) * K + i] != -1);
            }
            for (size_t h = 0; h < d.Hq; h++) {
                for (size_t j = 0; j < d.Dv; j++) {
                    bad_padding += static_cast<size_t>(attn[((b * d.Hq + h) * L + l) * d.Dv + j] != 0.0f);
                }
            }
        }
        EXPECT_EQ(bad_padding, 0u) << "left-padding rows must select nothing and output zeros";

        pair_outputs one;
        one.plan.sel.assign(all.plan.sel.begin() + (b * L + pad) * K, all.plan.sel.begin() + (b + 1) * L * K);
        one.plan.count.assign(all.plan.count.begin() + b * L + pad, all.plan.count.begin() + (b + 1) * L);
        one.attn.resize(ref.q_len * d.Hq * d.Dv);
        for (size_t t = 0; t < ref.q_len; t++) {
            for (size_t h = 0; h < d.Hq; h++) {
                for (size_t j = 0; j < d.Dv; j++) {
                    one.attn[(t * d.Hq + h) * d.Dv + j] = attn[((b * d.Hq + h) * L + pad + t) * d.Dv + j];
                }
            }
        }
        validate_pair(ref, one, atol, rtol);
    }
    validate_dense_state(net, in, refs);
}

/// One static dense network over `refs` (one batch row each), validated row by row.
void run_dense_rows(engine& engine, const std::vector<split_reference>& refs, const route& r, float atol, float rtol) {
    const auto in = dense_inputs_from_references(refs);
    auto net = build_dense_network(engine, in, Stage::Pair);
    net->execute();
    validate_dense_outputs(*net, in, refs, &r, atol, rtol);
}

void validate_and_run_pair(bool paged, engine& engine, const split_reference& ref, const route& r, float atol, float rtol) {
    if (!paged) {
        run_dense_rows(engine, {ref}, r, atol, rtol);
        return;
    }
    const auto gpu = run_paged_pair(engine, ref);
    expect_route(paged, gpu, r);
    validate_pair(ref, gpu, atol, rtol);
}

void run_pair_case(bool paged, const split_dims& d, size_t past_len, size_t q_len, Positions kind, const route& r, float rtol, float atol, float scale) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);
    const auto ref = make_split_reference(rg, d, past_len, q_len, kind, scale);
    validate_and_run_pair(paged, engine, ref, r, atol, rtol);
}

void run_pair_test(bool paged,
                   const split_dims& d,
                   size_t past_len,
                   size_t q_len,
                   Positions kind,
                   const route& r,
                   float rtol,
                   float atol = 2e-3f,
                   float scale = 1.0f) {
    if (!qsa_ops_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    ASSERT_GT((past_len + q_len) / d.r, d.block_topk) << "the selection must drop some complete blocks to be meaningful";
    run_pair_case(paged, d, past_len, q_len, kind, r, rtol, atol, scale);
}

/// Short context: every query sees at most block_topk complete blocks, so top-k selects all of them.
void run_full_selection_pair_test(bool paged, const split_dims& d, size_t past_len, size_t q_len, const route& r, float rtol, float atol = 2e-3f) {
    if (!qsa_ops_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    ASSERT_LE((past_len + q_len) / d.r, d.block_topk);
    run_pair_case(paged, d, past_len, q_len, Positions::Sequential, r, rtol, atol, 1.0f);
}

/// Independent references sharing the norm gains the network takes as single constants.
std::vector<split_reference> make_batch_references(const split_dims& d, const std::vector<std::pair<size_t, size_t>>& seqs) {
    tests::random_generator rg(GET_SUITE_NAME);
    std::vector<split_reference> refs;
    for (const auto& [past_len, q_len] : seqs) {
        refs.push_back(make_split_reference(rg, d, past_len, q_len, Positions::Sequential));
        refs.back().q_norm = refs.front().q_norm;
        refs.back().k_norm = refs.front().k_norm;
    }
    return refs;
}

/// One paged batch: each sequence gets its own contiguous run of pages.
paged_inputs pack_paged_inputs(const std::vector<split_reference>& refs) {
    paged_inputs in;
    in.block_indices_begins = {0};
    in.subsequence_begins = {0};
    for (size_t i = 0; i < refs.size(); i++) {
        auto one = paged_inputs_from_reference(refs[i]);
        if (i == 0 || one.d.max_positions > in.d.max_positions) {
            in.d = one.d;  // sequential positions: a longer rotary table is a superset
            in.rotary = one.rotary;
        }
        in.spp = one.spp;
        for (int32_t page : one.block_indices) {
            in.block_indices.push_back(page + static_cast<int32_t>(in.num_pages));
        }
        in.num_pages += one.num_pages;
        in.block_indices_begins.push_back(static_cast<int32_t>(in.num_pages));
        in.q_len += one.q_len;
        in.subsequence_begins.push_back(static_cast<int32_t>(in.q_len));
        in.past_lens.push_back(one.past_lens[0]);
        in.q_norm = one.q_norm;
        in.k_norm = one.k_norm;
        for (auto [dst, src] : {std::pair{&in.index_query, &one.index_query},
                                {&in.raw_index_keys, &one.raw_index_keys},
                                {&in.query, &one.query},
                                {&in.key, &one.key},
                                {&in.value, &one.value},
                                {&in.key_cache, &one.key_cache},
                                {&in.value_cache, &one.value_cache},
                                {&in.raw_k_cache, &one.raw_k_cache},
                                {&in.block_index_keys_cache, &one.block_index_keys_cache}}) {
            append(*dst, *src);
        }
        append(in.position_ids, one.position_ids);
        append(in.position_cache, one.position_cache);
    }
    in.qb_start = in.subsequence_begins;
    return in;
}

void validate_slices(bool paged, const std::vector<split_reference>& refs, const pair_outputs& all, const route& r, float atol, float rtol) {
    expect_route(paged, all, r);
    const auto& d = refs.front().d;
    size_t t0 = 0;
    for (const auto& ref : refs) {
        pair_outputs one;
        one.plan.sel.assign(all.plan.sel.begin() + t0 * d.block_topk, all.plan.sel.begin() + (t0 + ref.q_len) * d.block_topk);
        one.plan.count.assign(all.plan.count.begin() + t0, all.plan.count.begin() + t0 + ref.q_len);
        one.attn.assign(all.attn.begin() + t0 * d.Hq * d.Dv, all.attn.begin() + (t0 + ref.q_len) * d.Hq * d.Dv);
        SCOPED_TRACE("sequence past_len=" + std::to_string(ref.past_len) + " q_len=" + std::to_string(ref.q_len));
        validate_pair(ref, one, atol, rtol);
        t0 += ref.q_len;
    }
}

/// Multi-sequence paged batch: all decode (q_len == 1 each) or mixed decode/prefill.
void run_paged_batch_test(const split_dims& d, const std::vector<std::pair<size_t, size_t>>& seqs, const route& r, float rtol, float atol = 2e-3f) {
    if (!qsa_ops_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    auto& engine = get_test_engine();
    const auto refs = make_batch_references(d, seqs);
    auto net = build_paged_network(engine, pack_paged_inputs(refs), Stage::Pair);
    auto outputs = net->execute();
    pair_outputs all;
    all.attn = read_f16(outputs.at("attn").get_memory());
    read_pair_outputs(*net, all);
    validate_slices(true, refs, all, r, atol, rtol);
}

/// Dense batch, one row per (past_len, q_len). Rows may differ in key length (their K/V
/// are left-padded to the longest row); a row with q_len below the longest q_len must have
/// past_len == 0 and is left-padded.
void run_dense_batch_test(const split_dims& d, const std::vector<std::pair<size_t, size_t>>& rows, const route& r, float rtol, float atol = 2e-3f) {
    if (!qsa_ops_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    run_dense_rows(get_test_engine(), make_batch_references(d, rows), r, atol, rtol);
}

/// Dense batch of `batch` identical-length rows.
void run_dense_batch_test(const split_dims& d, size_t batch, size_t past_len, size_t q_len, const route& r, float rtol, float atol = 2e-3f) {
    run_dense_batch_test(d, std::vector<std::pair<size_t, size_t>>(batch, {past_len, q_len}), r, rtol, atol);
}

/// `full` restricted to its first `past_len + q_len` tokens, the last `q_len` being queries.
/// `full` must have been built with past_len == 0, so it holds every token's query.
split_reference slice_reference(const split_reference& full, size_t past_len, size_t q_len) {
    const auto& d = full.d;
    const size_t total = past_len + q_len;
    split_reference ref;
    ref.d = d;
    ref.past_len = past_len;
    ref.q_len = q_len;
    ref.positions.assign(full.positions.begin(), full.positions.begin() + total);
    ref.raw_index_keys_all.assign(full.raw_index_keys_all.begin(), full.raw_index_keys_all.begin() + total * d.Di);
    ref.index_query_cur.assign(full.index_query_cur.begin() + past_len * d.Hiq * d.Di, full.index_query_cur.begin() + total * d.Hiq * d.Di);
    ref.key_all.assign(full.key_all.begin(), full.key_all.begin() + total * d.Hkv * d.Dh);
    ref.value_all.assign(full.value_all.begin(), full.value_all.begin() + total * d.Hkv * d.Dv);
    ref.query_cur.assign(full.query_cur.begin() + past_len * d.Hq * d.Dh, full.query_cur.begin() + total * d.Hq * d.Dh);
    ref.q_norm = full.q_norm;
    ref.k_norm = full.k_norm;
    ref.rotary = full.rotary;
    return ref;
}

/// Functional state across calls: ONE dynamic dense network runs a sequence chunk by chunk,
/// and every call after the first consumes the state outputs (block_index_keys_cache, staged raw keys,
/// RoPE row) the previous call returned, as ReadValue/Assign would carry them.
void run_dense_state_chain_test(const split_dims& d, const std::vector<size_t>& q_lens, float rtol, float atol = 2e-3f) {
    if (!qsa_ops_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    using Out = ov::intel_gpu::op::QSAIndexer::OutputIdx;
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);
    const auto full = make_split_reference(rg, d, 0, std::accumulate(q_lens.begin(), q_lens.end(), size_t{0}), Positions::MRoPELike);
    std::unique_ptr<network> net;
    std::vector<float> summary, raw, rope;
    size_t summary_cap = 0;
    size_t past_len = 0;
    for (size_t q_len : q_lens) {
        SCOPED_TRACE("past_len=" + std::to_string(past_len) + " q_len=" + std::to_string(q_len));
        const auto ref = slice_reference(full, past_len, q_len);
        auto in = dense_inputs_from_reference(ref);
        if (net) {
            // Previous state, summary grown to this call's capacity (zero-filled tail).
            in.block_index_keys_cache.assign(in.summary_cap * d.Di, 0.0f);
            std::copy_n(summary.begin(), summary_cap * d.Di, in.block_index_keys_cache.begin());
            in.raw_state = raw;
            in.rope_state = rope;
            set_dense_inputs(engine, *net, in, Stage::Pair, true);
        } else {
            net = build_dense_network(engine, in, Stage::Pair, true);
        }
        net->execute();
        validate_dense_outputs(*net, in, {ref}, nullptr, atol, rtol);

        auto indexer = net->get_primitive("indexer");
        auto read_state = [&](size_t out) {
            auto values = read_f16(indexer->output_memory_ptr(out));
            values.resize(indexer->get_output_layout(out).count());
            return values;
        };
        summary = read_state(Out::BLOCK_INDEX_KEYS_CACHE);
        raw = read_state(Out::PAST_INCOMPLETE_RAW_K);
        rope = read_state(Out::PAST_INCOMPLETE_ROPE);
        summary_cap = in.summary_cap;
        past_len += q_len;
    }
}

// setenv/unsetenv are POSIX-only; MSVC provides _putenv_s, where an empty value removes the variable.
void set_env_var(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void unset_env_var(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

/// Caps the decode BLOCK_SCORES scratch so the Q2 chunk loop runs several times.
struct scoped_score_budget {
    explicit scoped_score_budget(const char* bytes) {
        set_env_var("OV_GPU_QSA_SCORE_BUDGET_BYTES", bytes);
    }
    ~scoped_score_budget() {
        unset_env_var("OV_GPU_QSA_SCORE_BUDGET_BYTES");
    }
};

const route decode_gqa{IndexerPath::Decode, AttnPath::GqaSplit};
const route dpas_gqa{IndexerPath::Dpas, AttnPath::GqaDpas};
const route dpas_dpas{IndexerPath::Dpas, AttnPath::Dpas};
const route decode_scalar{IndexerPath::Decode, AttnPath::Scalar};
const route decode_split{IndexerPath::Decode, AttnPath::Split};
const route fused_scalar{IndexerPath::Fused, AttnPath::Scalar};

// Decode batches: Q2 score_partition + top-k, Q3 GQA-DPAS (qwen38 dims).
TEST(qsa_ops_gpu_test, qwen38_paged_batch_decode_matches_reference) {
    run_paged_batch_test(qwen38_dims(), {{2100, 1}, {2600, 1}, {3071, 1}}, decode_gqa, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_dense_batch_decode_matches_reference) {
    run_dense_batch_test(qwen38_dims(), 3, 3071, 1, decode_gqa, 3e-3f);
}

// Same, with a one-row BLOCK_SCORES budget. The paged scratch row is sized from the page table
// (past_lens stay in GPU memory), i.e. the whole 1948-block pool, so only one row fits per
// chunk: 3 chunks of one sequence each.
TEST(qsa_ops_gpu_test, qwen38_paged_batch_decode_chunked_scores_match_reference) {
    scoped_score_budget budget("1");
    run_paged_batch_test(qwen38_dims(), {{2100, 1}, {2600, 1}, {3071, 1}}, {IndexerPath::Decode, AttnPath::GqaSplit, 3}, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_dense_batch_decode_chunked_scores_match_reference) {
    scoped_score_budget budget("1");
    run_dense_batch_test(qwen38_dims(), 3, 3071, 1, {IndexerPath::Decode, AttnPath::GqaSplit, 3}, 3e-3f);
}

// Budget of exactly two 768-block rows: a full chunk followed by a partial one.
TEST(qsa_ops_gpu_test, qwen38_dense_batch_decode_partial_last_chunk_matches_reference) {
    scoped_score_budget budget("6144");
    run_dense_batch_test(qwen38_dims(), 3, 3071, 1, {IndexerPath::Decode, AttnPath::GqaSplit, 2}, 3e-3f);
}

// Mixed batches (decode + prefill sequences in one call): dpas Q2 with qwen38 dims, the
// fused non-dpas Q2 with the toy dims (Di = 16 fails the dpas score gate).
TEST(qsa_ops_gpu_test, qwen38_paged_batch_mixed_matches_reference) {
    run_paged_batch_test(qwen38_dims(), {{2100, 1}, {1990, 64}, {2500, 1}}, dpas_gqa, 3e-3f);
}

TEST(qsa_ops_gpu_test, paged_batch_mixed_fused_score_topk_matches_reference) {
    run_paged_batch_test(split_dims{}, {{20, 1}, {22, 42}, {30, 1}}, fused_scalar, 2e-2f);
}

TEST(qsa_ops_gpu_test, paged_batch_decode_toy_dims_matches_reference) {
    run_paged_batch_test(split_dims{}, {{20, 1}, {37, 1}, {63, 1}}, decode_split, 2e-2f);
}

// Toy dims have no GQA-DPAS, so long decode takes the split + reduce fallback.
TEST(qsa_ops_gpu_test, paged_batch_long_decode_split_fallback_matches_reference) {
    run_paged_batch_test(split_dims{}, {{2100, 1}, {2500, 1}}, decode_split, 2e-2f);
}

TEST(qsa_ops_gpu_test, dense_batch_long_decode_split_fallback_matches_reference) {
    run_dense_batch_test(split_dims{}, 2, 2100, 1, decode_split, 2e-2f);
}

// ---------------------------------------------------------------------------
// Multi-batch dense rows of different key lengths: K/V of shorter rows are left-padded
// (logical key j at S - key_length + j), and rows shorter than L are fresh, left-padded
// sequences whose padding queries select nothing and output zeros. Padding storage is
// NaN, so any read of it fails the comparison.
// ---------------------------------------------------------------------------

TEST(qsa_ops_gpu_test, dense_batch_left_padding_fused_scalar_matches_reference) {
    run_dense_batch_test(split_dims{}, {{20, 44}, {0, 30}, {0, 44}, {0, 3}}, fused_scalar, 2e-2f);
}

TEST(qsa_ops_gpu_test, dense_batch_ragged_decode_scalar_matches_reference) {
    run_dense_batch_test(split_dims{}, {{63, 1}, {20, 1}, {37, 1}}, decode_scalar, 2e-2f);
}

TEST(qsa_ops_gpu_test, dense_batch_ragged_long_decode_split_matches_reference) {
    run_dense_batch_test(split_dims{}, {{2100, 1}, {15, 1}}, decode_split, 2e-2f);
}

// Padding spans whole and partial 16-query tiles of the plain DPAS kernel.
TEST(qsa_ops_gpu_test, dense_batch_left_padding_plain_dpas_matches_reference) {
    run_dense_batch_test(non_gqa_dpas_dims(), {{2100, 33}, {0, 9}, {0, 20}}, dpas_dpas, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_dense_batch_left_padding_prefill_matches_reference) {
    run_dense_batch_test(qwen38_dims(), {{4032, 64}, {0, 40}, {0, 64}}, dpas_gqa, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_dense_batch_ragged_decode_matches_reference) {
    run_dense_batch_test(qwen38_dims(), {{3071, 1}, {15, 1}, {2050, 1}}, decode_gqa, 3e-3f);
}

// ---------------------------------------------------------------------------
// Functional state across calls: each call consumes the state the previous call returned,
// including staged keys of an incomplete block that a later call completes.
// ---------------------------------------------------------------------------

TEST(qsa_ops_gpu_test, dense_state_carries_across_calls) {
    run_dense_state_chain_test(split_dims{}, {37, 1, 1, 6, 1, 3, 2}, 2e-2f);
}

TEST(qsa_ops_gpu_test, qwen38_dense_state_carries_across_calls) {
    run_dense_state_chain_test(qwen38_dims(), {2050, 1, 1, 17, 1}, 3e-3f);
}

// Toy Dh = 16 is below the 64-byte DPAS row minimum, so prefill takes the scalar Q3 kernel.
TEST(qsa_ops_gpu_test, paged_pair_matches_reference) {
    run_pair_test(true, split_dims{}, 20, 44, Positions::Sequential, fused_scalar, 2e-2f);
}

TEST(qsa_ops_gpu_test, dense_pair_matches_reference) {
    run_pair_test(false, split_dims{}, 20, 44, Positions::Sequential, fused_scalar, 2e-2f);
}

TEST(qsa_ops_gpu_test, paged_pair_mrope_positions_match_reference) {
    run_pair_test(true, split_dims{}, 22, 42, Positions::MRoPELike, fused_scalar, 2e-2f);
}

TEST(qsa_ops_gpu_test, dense_pair_mrope_positions_match_reference) {
    run_pair_test(false, split_dims{}, 22, 42, Positions::MRoPELike, fused_scalar, 2e-2f);
}

// Qwen3.8-Flash-Next shapes: decode one token over a long history (partial last block
// completed by this call), and a 64-token prefill chunk on top of a cached prefix.
TEST(qsa_ops_gpu_test, qwen38_paged_decode_matches_reference) {
    run_pair_test(true, qwen38_dims(), 8191, 1, Positions::Sequential, decode_gqa, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_paged_prefill_matches_reference) {
    run_pair_test(true, qwen38_dims(), 4032, 64, Positions::Sequential, dpas_gqa, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_paged_prefill_mrope_matches_reference) {
    run_pair_test(true, qwen38_dims(), 4030, 66, Positions::MRoPELike, dpas_gqa, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_dense_decode_matches_reference) {
    run_pair_test(false, qwen38_dims(), 8191, 1, Positions::Sequential, decode_gqa, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_dense_prefill_matches_reference) {
    run_pair_test(false, qwen38_dims(), 4032, 64, Positions::Sequential, dpas_gqa, 3e-3f);
}

// Prefill rows on both sides of the top-k boundary (510..514 complete blocks vs K = 512):
// one top-k dispatch mixes select-all rows with real selections.
TEST(qsa_ops_gpu_test, qwen38_paged_prefill_crossing_topk_boundary_matches_reference) {
    run_pair_test(true, qwen38_dims(), 2040, 17, Positions::Sequential, dpas_gqa, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_dense_prefill_crossing_topk_boundary_matches_reference) {
    run_pair_test(false, qwen38_dims(), 2040, 17, Positions::Sequential, dpas_gqa, 3e-3f);
}

// ---------------------------------------------------------------------------
// Short context: at most block_topk complete blocks, so the top-k selects every block
// without reading scores and early tokens attend to the causal tail only.
// ---------------------------------------------------------------------------

TEST(qsa_ops_gpu_test, qwen38_paged_short_prefill_selects_every_block) {
    run_full_selection_pair_test(true, qwen38_dims(), 0, 17, dpas_gqa, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_dense_short_prefill_selects_every_block) {
    run_full_selection_pair_test(false, qwen38_dims(), 0, 17, dpas_gqa, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_paged_short_decode_selects_every_block) {
    run_full_selection_pair_test(true, qwen38_dims(), 15, 1, decode_gqa, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_dense_short_decode_selects_every_block) {
    run_full_selection_pair_test(false, qwen38_dims(), 15, 1, decode_gqa, 3e-3f);
}

// One decode batch with 4, exactly 512 (score bypass) and 513 (scored) complete blocks.
TEST(qsa_ops_gpu_test, qwen38_paged_batch_decode_around_topk_boundary_matches_reference) {
    run_paged_batch_test(qwen38_dims(), {{15, 1}, {2047, 1}, {2051, 1}}, decode_gqa, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_dense_batch_decode_at_topk_boundary_matches_reference) {
    run_dense_batch_test(qwen38_dims(), 2, 2047, 1, decode_gqa, 3e-3f);
}

// Short decode, a long decode and a fresh prefill sequence in one call.
TEST(qsa_ops_gpu_test, qwen38_paged_batch_mixed_short_and_long_matches_reference) {
    run_paged_batch_test(qwen38_dims(), {{15, 1}, {2100, 1}, {0, 17}}, dpas_gqa, 3e-3f);
}

// ---------------------------------------------------------------------------
// Plain (non-GQA) DPAS attention: dims that fail the GQA-DPAS gate but fit plain DPAS.
// q_len = 17 leaves a one-token last query tile; q_len = 16 is exactly one tile.
// ---------------------------------------------------------------------------

TEST(qsa_ops_gpu_test, paged_pair_plain_dpas_attention_matches_reference) {
    run_pair_test(true, non_gqa_dpas_dims(), 2100, 17, Positions::Sequential, dpas_dpas, 3e-3f);
}

TEST(qsa_ops_gpu_test, dense_pair_plain_dpas_attention_matches_reference) {
    run_pair_test(false, non_gqa_dpas_dims(), 2100, 17, Positions::Sequential, dpas_dpas, 3e-3f);
}

TEST(qsa_ops_gpu_test, paged_pair_plain_dpas_four_workers_matches_reference) {
    run_pair_test(true, wide_group_dims(), 2100, 16, Positions::Sequential, dpas_dpas, 3e-3f);
}

TEST(qsa_ops_gpu_test, dense_pair_plain_dpas_four_workers_matches_reference) {
    run_pair_test(false, wide_group_dims(), 2100, 16, Positions::Sequential, dpas_dpas, 3e-3f);
}

TEST(qsa_ops_gpu_test, paged_pair_plain_dpas_one_head_per_tile_matches_reference) {
    run_pair_test(true, mha_dpas_dims(), 2100, 17, Positions::Sequential, dpas_dpas, 3e-3f);
}

TEST(qsa_ops_gpu_test, dense_pair_plain_dpas_one_head_per_tile_matches_reference) {
    run_pair_test(false, mha_dpas_dims(), 2100, 17, Positions::Sequential, dpas_dpas, 3e-3f);
}

// Per-sequence query tiles: two single-token decode tiles plus a 16 + 1 prefill.
TEST(qsa_ops_gpu_test, paged_batch_mixed_plain_dpas_attention_matches_reference) {
    run_paged_batch_test(non_gqa_dpas_dims(), {{15, 1}, {2100, 1}, {2040, 17}}, dpas_dpas, 3e-3f);
}

// ---------------------------------------------------------------------------
// Decode split-KV threshold for non-GQA dims. Paged past_lens are GPU-resident, so it always
// splits; dense compares the whole K length (past_len + 1) with 2048.
// ---------------------------------------------------------------------------

TEST(qsa_ops_gpu_test, paged_batch_short_decode_uses_split) {
    run_paged_batch_test(non_gqa_dpas_dims(), {{2047, 1}, {3000, 1}}, decode_split, 3e-3f);
}

TEST(qsa_ops_gpu_test, paged_batch_long_decode_uses_split) {
    run_paged_batch_test(non_gqa_dpas_dims(), {{2048, 1}, {3000, 1}}, decode_split, 3e-3f);
}

TEST(qsa_ops_gpu_test, dense_batch_decode_below_split_threshold_uses_scalar) {
    run_dense_batch_test(non_gqa_dpas_dims(), 2, 2046, 1, decode_scalar, 3e-3f);
}

TEST(qsa_ops_gpu_test, dense_batch_decode_at_split_threshold_uses_split) {
    run_dense_batch_test(non_gqa_dpas_dims(), 2, 2047, 1, decode_split, 3e-3f);
}

// ---------------------------------------------------------------------------
// Dedicated numerical-precision stress tests. `validate_pair()` always reports
// max_abs/max_rel error via a `QSA_PRECISION` log line (grep the test output) and
// asserts finiteness, in addition to the combined absolute+relative bound below.
// ---------------------------------------------------------------------------

TEST(qsa_ops_gpu_test, paged_pair_large_magnitude_inputs_stay_finite_and_accurate) {
    // Activations scaled to +-3 (vs. the usual +-1): dot products over Dh=256/Di=128
    // stay well inside the fp16 range (max 65504), exercising RMSNorm/RoPE/softmax at
    // several times the usual magnitude. The looser bound (vs. the other pair tests) is
    // deliberate: softmax is EXPONENTIALLY sensitive to logit differences, so at a larger
    // scale (bigger typical logits) the same small fp32-vs-double accumulation-order
    // difference between the GPU's DPAS reduction and this reference's summation gets
    // amplified into a visibly larger, but still small, output difference for a minority
    // of elements with closely competing logits -- this is expected floating-point
    // behavior, not a kernel bug, and a tighter bound here would be testing summation
    // order, not correctness.
    run_pair_test(true, qwen38_dims(), 4032, 64, Positions::Sequential, dpas_gqa, /*rtol=*/1e-2f, /*atol=*/1e-2f, /*scale=*/3.0f);
}

TEST(qsa_ops_gpu_test, dense_pair_large_magnitude_inputs_stay_finite_and_accurate) {
    run_pair_test(false, qwen38_dims(), 4032, 64, Positions::Sequential, dpas_gqa, /*rtol=*/1e-2f, /*atol=*/1e-2f, /*scale=*/3.0f);
}

TEST(qsa_ops_gpu_test, paged_pair_dominant_outlier_component_stays_accurate) {
    // One channel 40x the scale of the rest in both the first query token's index head
    // and the earliest historical index key: stresses RMSNorm's sum-of-squares reduction
    // (a single term dominating the accumulation) rather than merely scaling everything.
    if (!qsa_ops_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    const auto d = qwen38_dims();
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);
    auto ref = make_split_reference(rg, d, 4032, 64, Positions::Sequential);
    ref.index_query_cur[0] = static_cast<float>(ov::float16(40.0f));
    ref.raw_index_keys_all[0] = static_cast<float>(ov::float16(40.0f));
    ref.summary_memo.clear();
    validate_and_run_pair(true, engine, ref, dpas_gqa, /*atol=*/4e-3f, /*rtol=*/5e-3f);
}

TEST(qsa_ops_gpu_test, dense_pair_dominant_outlier_component_stays_accurate) {
    if (!qsa_ops_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    const auto d = qwen38_dims();
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);
    auto ref = make_split_reference(rg, d, 4032, 64, Positions::Sequential);
    ref.index_query_cur[0] = static_cast<float>(ov::float16(40.0f));
    ref.raw_index_keys_all[0] = static_cast<float>(ov::float16(40.0f));
    ref.summary_memo.clear();
    validate_and_run_pair(false, engine, ref, dpas_gqa, /*atol=*/4e-3f, /*rtol=*/5e-3f);
}

// Many more blocks than the prefill cases above: checks that per-block summary error and
// online-softmax accumulation error do not grow with the number of terms summed.
TEST(qsa_ops_gpu_test, qwen38_paged_long_context_accumulation_matches_reference) {
    run_pair_test(true, qwen38_dims(), 8192, 128, Positions::Sequential, dpas_gqa, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_dense_long_context_accumulation_matches_reference) {
    run_pair_test(false, qwen38_dims(), 8192, 128, Positions::Sequential, dpas_gqa, 3e-3f);
}

// ---------------------------------------------------------------------------
// Dynamic shapes: ONE network built with dynamic layouts is executed for several
// (past_len, q_len). Every call must reuse the same shape-agnostic impls (no per-shape
// recompilation), pick its route at runtime and match the reference.
// ---------------------------------------------------------------------------

struct dynamic_step {
    size_t past_len;
    size_t q_len;
    route r;
};

void run_dynamic_pair_test(bool paged, const split_dims& d, const std::vector<dynamic_step>& steps, float rtol, float atol = 2e-3f) {
    if (!qsa_ops_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);
    std::unique_ptr<network> net;
    const primitive_impl* indexer_impl = nullptr;
    const primitive_impl* attn_impl = nullptr;
    for (const auto& step : steps) {
        SCOPED_TRACE("past_len=" + std::to_string(step.past_len) + " q_len=" + std::to_string(step.q_len));
        const auto ref = make_split_reference(rg, d, step.past_len, step.q_len, Positions::Sequential);
        dense_inputs dense_in;
        if (paged) {
            const auto in = paged_inputs_from_reference(ref);
            if (net) {
                set_paged_inputs(engine, *net, in, Stage::Pair, true);
            } else {
                net = build_paged_network(engine, in, Stage::Pair, true);
            }
        } else {
            dense_in = dense_inputs_from_reference(ref);
            if (net) {
                set_dense_inputs(engine, *net, dense_in, Stage::Pair, true);
            } else {
                net = build_dense_network(engine, dense_in, Stage::Pair, true);
            }
        }
        auto outputs = net->execute();
        pair_outputs gpu;
        auto attn = read_f16(outputs.at("attn").get_memory());
        attn.resize(net->get_primitive("attn")->get_output_layout().count());
        gpu.attn = paged ? attn : bhld_to_token_major(attn, ref.d, ref.q_len);
        read_pair_outputs(*net, gpu);

        const auto* indexer = net->get_primitive("indexer")->get_impl();
        const auto* attention = net->get_primitive("attn")->get_impl();
        ASSERT_TRUE(indexer->is_dynamic());
        ASSERT_TRUE(attention->is_dynamic());
        if (indexer_impl == nullptr) {
            indexer_impl = indexer;
            attn_impl = attention;
        }
        EXPECT_EQ(indexer, indexer_impl) << "a new shape must reuse the shape-agnostic indexer impl";
        EXPECT_EQ(attention, attn_impl) << "a new shape must reuse the shape-agnostic attention impl";
        if (!paged) {
            validate_dense_outputs(*net, dense_in, {ref}, &step.r, atol, rtol);
            continue;
        }
        expect_route(paged, gpu, step.r);
        validate_pair(ref, gpu, atol, rtol);
    }
}

// Paged past_lens live in GPU memory, so its non-GQA decode always takes split + reduce;
// dense gates the split on the (shape-known) K length instead.
std::vector<dynamic_step> toy_dynamic_steps(bool paged) {
    const route decode = paged ? decode_split : decode_scalar;
    return {{0, 37, fused_scalar}, {37, 1, decode}, {38, 1, decode}, {39, 6, fused_scalar}, {45, 1, decode}};
}
const std::vector<dynamic_step> qwen38_dynamic_steps = {{2100, 17, dpas_gqa}, {2117, 1, decode_gqa}, {2118, 1, decode_gqa}, {40, 9, dpas_gqa}};

TEST(qsa_ops_gpu_test, paged_pair_dynamic_shapes_reuse_one_network) {
    run_dynamic_pair_test(true, split_dims{}, toy_dynamic_steps(true), 2e-3f);
}

TEST(qsa_ops_gpu_test, dense_pair_dynamic_shapes_reuse_one_network) {
    run_dynamic_pair_test(false, split_dims{}, toy_dynamic_steps(false), 2e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_paged_pair_dynamic_shapes_reuse_one_network) {
    run_dynamic_pair_test(true, qwen38_dims(), qwen38_dynamic_steps, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_dense_pair_dynamic_shapes_reuse_one_network) {
    run_dynamic_pair_test(false, qwen38_dims(), qwen38_dynamic_steps, 3e-3f);
}

// ---------------------------------------------------------------------------
// SparseSDPA over K/V buffers whose sequence axis is padded past the logical history,
// as a KVCache producer hands them over: the kernels must step by the allocated pitch.
// ---------------------------------------------------------------------------

/// [1, Hkv, S + pad, D] f16 memory holding token-major `src` ([S, Hkv*D]); padded rows are NaN
/// so any read through the logical (unpadded) pitch poisons the output.
memory::ptr make_seq_padded_f16(engine& eng, const std::vector<float>& src, size_t hkv, size_t s, size_t dim, size_t pad) {
    const layout l{ov::PartialShape{1, I(hkv), I(s), I(dim)}, data_types::f16, format::bfyx, padding({0, 0, 0, 0}, {0, 0, static_cast<int32_t>(pad), 0})};
    std::vector<ov::float16> host(hkv * (s + pad) * dim, ov::float16(std::numeric_limits<float>::quiet_NaN()));
    for (size_t h = 0; h < hkv; h++) {
        for (size_t p = 0; p < s; p++) {
            for (size_t j = 0; j < dim; j++) {
                host[(h * (s + pad) + p) * dim + j] = ov::float16(src[p * hkv * dim + h * dim + j]);
            }
        }
    }
    auto mem = eng.allocate_memory(l, allocation_type::usm_device);
    mem->copy_from(get_test_stream(), host.data(), true);
    return mem;
}

std::vector<std::vector<size_t>> plan_selections(const selection_plan& plan, size_t q_len, size_t topk) {
    std::vector<std::vector<size_t>> out(q_len);
    for (size_t t = 0; t < q_len; t++) {
        for (int32_t i = 0; i < plan.count[t]; i++) {
            out[t].push_back(static_cast<size_t>(plan.sel[t * topk + i]));
        }
    }
    return out;
}

void run_padded_kv_attention_test(const split_dims& d,
                                  size_t past_len,
                                  size_t q_len,
                                  size_t k_pad,
                                  size_t v_pad,
                                  AttnPath path,
                                  float rtol,
                                  float atol = 2e-3f) {
    if (!qsa_ops_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);
    const auto ref = make_split_reference(rg, d, past_len, q_len, Positions::Sequential);
    const auto total = ref.total_len();
    const auto plan = synthetic_plan(d, past_len, q_len);
    const auto in = dense_inputs_from_reference(ref);

    auto key = make_seq_padded_f16(engine, ref.key_all, d.Hkv, total, d.Dh, k_pad);
    auto value = make_seq_padded_f16(engine, ref.value_all, d.Hkv, total, d.Dv, v_pad);

    ov::intel_gpu::op::SparseSDPA::Config cfg;
    cfg.num_heads = d.Hq;
    cfg.num_kv_heads = d.Hkv;
    cfg.num_selection_heads = 1;
    cfg.k_head_size = d.Dh;
    cfg.v_head_size = d.Dv;
    cfg.key_block_size = d.r;
    cfg.query_block_size = 1;
    cfg.causal = true;

    topology topo;
    topo.add(input_layout("query", layout{{1, I(d.Hq), I(q_len), I(d.Dh)}, data_types::f16, format::bfyx}));
    topo.add(input_layout("key", key->get_layout()));
    topo.add(input_layout("value", value->get_layout()));
    topo.add(data("sel", make_i32(engine, {1, 1, I(q_len), I(d.block_topk)}, plan.sel)));
    topo.add(data("count", make_i32(engine, {1, 1, I(q_len)}, plan.count)));
    topo.add(data("key_length", make_i32(engine, {1}, {static_cast<int32_t>(total)})));
    topo.add(sparse_sdpa("attn",
                         {input_info("query"), input_info("key"), input_info("value"), input_info("sel"), input_info("count"), input_info("key_length")},
                         cfg,
                         1.0f / std::sqrt(static_cast<float>(d.Dh))));
    network net(engine, topo, make_config(engine));
    net.set_input_data("query", make_f16(engine, {1, I(d.Hq), I(q_len), I(d.Dh)}, in.query_bhld));
    net.set_input_data("key", key);
    net.set_input_data("value", value);
    auto outputs = net.execute();

    const auto* const attn_params = net.get_primitive("attn")->get_impl_params();
    ASSERT_TRUE(static_cast<bool>(attn_params->get_input_layout(SparseSDPAIn::KEY).data_padding)) << "SparseSDPA must see the padded key buffer";
    EXPECT_EQ(executed_kernels(net, "attn"), expected_attention_kernels(false, resolve_route(route{IndexerPath::Decode, path}, false, total)));
    const auto gpu = bhld_to_token_major(read_f16(outputs.at("attn").get_memory()), d, q_len);
    const auto expected = ref.attention(plan_selections(plan, q_len, d.block_topk));
    const auto stats = compute_error_stats(gpu, expected);
    print_error_stats("padded_kv_attention", stats);
    ASSERT_EQ(stats.non_finite, 0u) << "a padded K/V row was read: the kernel ignored the allocated sequence pitch";
    size_t mismatches = 0;
    for (size_t i = 0; i < expected.size(); i++) {
        if (std::fabs(gpu[i] - expected[i]) > atol + rtol * std::fabs(expected[i]) && mismatches++ < 8) {
            ADD_FAILURE() << "attn[" << i << "] gpu=" << gpu[i] << " ref=" << expected[i];
        }
    }
    EXPECT_EQ(mismatches, 0u);
}

/// Two KV heads, so a wrong per-head pitch lands in another head's (or padded) rows.
split_dims two_kv_head_dims() {
    split_dims d;
    d.Hq = 4;
    d.Hkv = 2;
    return d;
}

TEST(qsa_ops_gpu_test, dense_attention_scalar_reads_padded_kv) {
    run_padded_kv_attention_test(two_kv_head_dims(), 37, 5, 11, 7, AttnPath::Scalar, 2e-3f);
}

TEST(qsa_ops_gpu_test, dense_attention_split_reads_padded_kv) {
    run_padded_kv_attention_test(two_kv_head_dims(), 2050, 1, 128, 64, AttnPath::Split, 2e-3f);
}

TEST(qsa_ops_gpu_test, dense_attention_plain_dpas_reads_padded_kv) {
    auto d = non_gqa_dpas_dims();
    d.Hq = 8;
    d.Hkv = 2;
    run_padded_kv_attention_test(d, 37, 20, 13, 5, AttnPath::Dpas, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_dense_attention_gqa_dpas_reads_padded_kv) {
    run_padded_kv_attention_test(qwen38_dims(), 37, 5, 9, 17, AttnPath::GqaDpas, 3e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_dense_attention_gqa_split_reads_padded_kv) {
    run_padded_kv_attention_test(qwen38_dims(), 40, 1, 9, 17, AttnPath::GqaSplit, 3e-3f);
}

// ---------------------------------------------------------------------------
// End to end through ov::Core: SparseSDPA reads K/V straight from KVCache nodes
// (ReadValue -> Concat -> Assign), one compiled model, several infer() calls with growing
// history. KVCache pre-allocates ahead on the sequence axis, i.e. hands over padded K/V.
// ---------------------------------------------------------------------------

void run_sparse_sdpa_kv_cache_model_test(const split_dims& d, const std::vector<size_t>& q_lens, float rtol, float atol = 2e-3f) {
    if (!qsa_ops_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    ov::Core core;
    const auto f16 = ov::element::f16;
    auto query = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{1, I(d.Hq), -1, I(d.Dh)});
    auto key_new = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{1, I(d.Hkv), -1, I(d.Dh)});
    auto value_new = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{1, I(d.Hkv), -1, I(d.Dv)});
    auto sel = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1, -1, I(d.block_topk)});
    auto count = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1, -1});
    auto key_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

    ov::SinkVector sinks;
    auto kv_cache = [&](const std::shared_ptr<ov::op::v0::Parameter>& fresh, size_t head_size, const std::string& name) {
        auto var = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{ov::PartialShape{1, I(d.Hkv), -1, I(head_size)}, f16, name});
        auto past = std::make_shared<ov::op::v6::ReadValue>(var);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past, fresh}, 2);
        sinks.push_back(std::make_shared<ov::op::v6::Assign>(concat, var));
        return concat;
    };

    ov::intel_gpu::op::SparseSDPA::Config cfg;
    cfg.num_heads = d.Hq;
    cfg.num_kv_heads = d.Hkv;
    cfg.num_selection_heads = 1;
    cfg.k_head_size = d.Dh;
    cfg.v_head_size = d.Dv;
    cfg.key_block_size = d.r;
    cfg.query_block_size = 1;
    cfg.causal = true;
    auto attn = std::make_shared<ov::intel_gpu::op::SparseSDPA>(
        ov::OutputVector{query, kv_cache(key_new, d.Dh, "k_cache"), kv_cache(value_new, d.Dv, "v_cache"), sel, count, key_length},
        cfg);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(attn)},
                                             sinks,
                                             ov::ParameterVector{query, key_new, value_new, sel, count, key_length});
    auto compiled = core.compile_model(model, "GPU", ov::hint::inference_precision(f16));
    auto request = compiled.create_infer_request();

    tests::random_generator rg(GET_SUITE_NAME);
    split_reference ref;
    ref.d = d;
    for (size_t q_len : q_lens) {
        SCOPED_TRACE("past_len=" + std::to_string(ref.total_len()) + " q_len=" + std::to_string(q_len));
        ref.past_len = ref.total_len();
        ref.q_len = q_len;
        ref.query_cur = random_vec(rg, q_len * d.Hq * d.Dh, 1.0f);
        auto key_cur = random_vec(rg, q_len * d.Hkv * d.Dh, 1.0f);
        auto value_cur = random_vec(rg, q_len * d.Hkv * d.Dv, 1.0f);
        for (auto* v : {&ref.query_cur, &key_cur, &value_cur}) {
            quantize_f16(*v);
        }
        append(ref.key_all, key_cur);
        append(ref.value_all, value_cur);
        const auto plan = synthetic_plan(d, ref.past_len, q_len);

        // Token-major [L, H*D] -> [1, H, L, D].
        auto to_bhld = [q_len](const std::vector<float>& src, size_t heads, size_t dim) {
            ov::Tensor t(ov::element::f16, ov::Shape{1, heads, q_len, dim});
            auto* dst = t.data<ov::float16>();
            for (size_t l = 0; l < q_len; l++) {
                for (size_t h = 0; h < heads; h++) {
                    for (size_t j = 0; j < dim; j++) {
                        dst[(h * q_len + l) * dim + j] = ov::float16(src[(l * heads + h) * dim + j]);
                    }
                }
            }
            return t;
        };
        request.set_tensor(query, to_bhld(ref.query_cur, d.Hq, d.Dh));
        request.set_tensor(key_new, to_bhld(key_cur, d.Hkv, d.Dh));
        request.set_tensor(value_new, to_bhld(value_cur, d.Hkv, d.Dv));
        ov::Tensor sel_t(ov::element::i32, ov::Shape{1, 1, q_len, d.block_topk});
        std::copy(plan.sel.begin(), plan.sel.end(), sel_t.data<int32_t>());
        ov::Tensor count_t(ov::element::i32, ov::Shape{1, 1, q_len});
        std::copy(plan.count.begin(), plan.count.end(), count_t.data<int32_t>());
        request.set_tensor(sel, sel_t);
        request.set_tensor(count, count_t);
        ov::Tensor key_length_t(ov::element::i32, ov::Shape{1});
        key_length_t.data<int32_t>()[0] = static_cast<int32_t>(ref.total_len());
        request.set_tensor(key_length, key_length_t);
        request.infer();

        const auto out = request.get_output_tensor(0);
        ASSERT_EQ(out.get_shape(), (ov::Shape{1, d.Hq, q_len, d.Dv}));
        const auto* out_data = out.data<ov::float16>();
        const auto gpu = bhld_to_token_major(std::vector<float>(out_data, out_data + out.get_size()), d, q_len);
        const auto expected = ref.attention(plan_selections(plan, q_len, d.block_topk));
        const auto stats = compute_error_stats(gpu, expected);
        print_error_stats("kv_cache_model_attention", stats);
        ASSERT_EQ(stats.non_finite, 0u);
        size_t mismatches = 0;
        for (size_t i = 0; i < expected.size(); i++) {
            if (std::fabs(gpu[i] - expected[i]) > atol + rtol * std::fabs(expected[i]) && mismatches++ < 8) {
                ADD_FAILURE() << "attn[" << i << "] gpu=" << gpu[i] << " ref=" << expected[i];
            }
        }
        ASSERT_EQ(mismatches, 0u);
    }
}

TEST(qsa_ops_gpu_test, sparse_sdpa_reads_kv_cache_across_infer_calls) {
    run_sparse_sdpa_kv_cache_model_test(two_kv_head_dims(), {37, 1, 1, 6, 1, 2003, 1, 1}, 2e-3f);
}

TEST(qsa_ops_gpu_test, qwen38_sparse_sdpa_reads_kv_cache_across_infer_calls) {
    run_sparse_sdpa_kv_cache_model_test(qwen38_dims(), {37, 1, 1, 9, 1}, 3e-3f);
}

// ---------------------------------------------------------------------------
// Host-level op validation (no GPU required)
// ---------------------------------------------------------------------------

std::shared_ptr<ov::op::v0::Parameter> make_param(const ov::PartialShape& shape, ov::element::Type_t et = ov::element::f16) {
    return std::make_shared<ov::op::v0::Parameter>(et, shape);
}

ov::OutputVector make_paged_indexer_args(const split_dims& d, size_t num_pages, size_t summaries_per_page) {
    return {make_param({-1, I(d.Hiq * d.Di)}),
            make_param({-1, I(d.Di)}),
            make_param({-1, I(d.rotary)}),
            make_param({-1, I(d.rotary)}),
            make_param({I(d.Di)}),
            make_param({I(d.Di)}),
            make_param({-1}, ov::element::i32),
            make_param({-1}, ov::element::i32),
            make_param({-1}, ov::element::i32),
            make_param({-1}, ov::element::i32),
            make_param({I(num_pages), 1, I(d.page_size), I(d.Di)}),
            make_param({I(num_pages), I(summaries_per_page), I(2 * d.rotary)}),
            make_param({I(num_pages), 1, I(summaries_per_page), I(d.Di)})};
}

ov::OutputVector make_dense_indexer_args(const split_dims& d, size_t h_ik = 1) {
    return {make_param({1, -1, I(d.Hiq), I(d.Di)}),
            make_param({1, -1, I(h_ik), I(d.Di)}),
            make_param({1, -1, I(d.rotary)}),
            make_param({1, -1, I(d.rotary)}),
            make_param({1}, ov::element::i32),
            make_param({I(d.Di)}),
            make_param({I(d.Di)}),
            make_param({1, I(h_ik), -1, I(d.Di)}),
            make_param({1, I(h_ik), I(d.r - 1), I(d.Di)}),
            make_param({1, I(2 * d.rotary)})};
}

ov::intel_gpu::op::PagedQSAIndexer::Config make_paged_indexer_config(const split_dims& d) {
    ov::intel_gpu::op::PagedQSAIndexer::Config c;
    c.key_block_size = d.r;
    c.block_topk = d.block_topk;
    c.num_selection_heads = 1;
    c.indexer_rotary_dim = d.rotary;
    c.indexer_head_dim = d.Di;
    c.epsilon = d.eps;
    c.scale = 1.0f / std::sqrt(static_cast<float>(d.Di));
    return c;
}

ov::intel_gpu::op::QSAIndexer::Config make_dense_indexer_config(const split_dims& d) {
    ov::intel_gpu::op::QSAIndexer::Config c;
    c.key_block_size = d.r;
    c.block_topk = d.block_topk;
    c.num_selection_heads = 1;
    c.indexer_rotary_dim = d.rotary;
    c.indexer_head_dim = d.Di;
    c.epsilon = d.eps;
    c.scale = 1.0f / std::sqrt(static_cast<float>(d.Di));
    return c;
}

TEST(qsa_ops_gpu_test, paged_qsa_indexer_config_validation_rejects_illegal_attributes) {
    std::string reason;
    split_dims d;
    auto base = make_paged_indexer_config(d);
    ASSERT_TRUE(ov::intel_gpu::op::PagedQSAIndexer::validate_config(base, reason)) << reason;

    auto c = base;
    c.key_block_size = 0;
    ASSERT_FALSE(ov::intel_gpu::op::PagedQSAIndexer::validate_config(c, reason));

    c = base;
    c.block_topk = 0;
    ASSERT_FALSE(ov::intel_gpu::op::PagedQSAIndexer::validate_config(c, reason));

    c = base;
    c.indexer_rotary_dim = c.indexer_head_dim + 2;
    ASSERT_FALSE(ov::intel_gpu::op::PagedQSAIndexer::validate_config(c, reason));

    c = base;
    c.scale = -1.0f;
    ASSERT_FALSE(ov::intel_gpu::op::PagedQSAIndexer::validate_config(c, reason));

    // scale == 0 selects the 1 / sqrt(indexer_head_dim) default.
    c = base;
    c.scale = 0.0f;
    ASSERT_TRUE(ov::intel_gpu::op::PagedQSAIndexer::validate_config(c, reason)) << reason;
    EXPECT_FLOAT_EQ(ov::intel_gpu::op::PagedQSAIndexer::effective_scale(c), 1.0f / std::sqrt(static_cast<float>(d.Di)));
}

TEST(qsa_ops_gpu_test, paged_qsa_indexer_rejects_page_size_not_divisible_by_key_block_size) {
    split_dims d;
    using In = ov::intel_gpu::op::PagedQSAIndexer::InputIdx;
    auto args = make_paged_indexer_args(d, 8, d.page_size / d.r);
    args[In::INDEX_KEY_CACHE] = make_param({8, 1, I(d.page_size + 1), I(d.Di)});  // page_size is read from this cache
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::PagedQSAIndexer>(args, make_paged_indexer_config(d)), ov::Exception);
}

TEST(qsa_ops_gpu_test, paged_qsa_indexer_output_shapes_and_clone) {
    split_dims d;
    const size_t num_pages = 8;
    const size_t summaries_per_page = d.page_size / d.r;
    auto node = std::make_shared<ov::intel_gpu::op::PagedQSAIndexer>(make_paged_indexer_args(d, num_pages, summaries_per_page), make_paged_indexer_config(d));

    ASSERT_EQ(node->get_output_size(), 2u);
    const auto& selected_block_indices_ps = node->get_output_partial_shape(0);
    ASSERT_EQ(selected_block_indices_ps.size(), 3u);
    ASSERT_EQ(selected_block_indices_ps[1].get_length(), 1);  // num_selection_heads == 1
    ASSERT_EQ(selected_block_indices_ps[2].get_length(), I(d.block_topk));
    ASSERT_EQ(node->get_output_element_type(0), ov::element::i32);
    ASSERT_EQ(node->get_output_element_type(1), ov::element::i32);

    auto clone = node->clone_with_new_inputs(node->input_values());
    ASSERT_NE(std::dynamic_pointer_cast<ov::intel_gpu::op::PagedQSAIndexer>(clone), nullptr)
        << "clone_with_new_inputs() must not silently downgrade to the base ov::op::internal::PagedQSAIndexer type";
}

TEST(qsa_ops_gpu_test, paged_qsa_indexer_resolves_default_scale) {
    split_dims d;
    auto cfg = make_paged_indexer_config(d);
    cfg.scale = 0.0f;
    auto node = std::make_shared<ov::intel_gpu::op::PagedQSAIndexer>(make_paged_indexer_args(d, 8, d.page_size / d.r), cfg);
    EXPECT_FLOAT_EQ(node->get_config().scale, 0.0f);  // sentinel stays in the IR; consumers resolve it
    EXPECT_FLOAT_EQ(ov::intel_gpu::op::PagedQSAIndexer::effective_scale(node->get_config()), 1.0f / std::sqrt(static_cast<float>(d.Di)));
}

TEST(qsa_ops_gpu_test, paged_qsa_indexer_rejects_wrong_input_count) {
    split_dims d;
    auto args = make_paged_indexer_args(d, 8, d.page_size / d.r);
    args.pop_back();
    ASSERT_THROW((void)std::make_shared<ov::intel_gpu::op::PagedQSAIndexer>(args, make_paged_indexer_config(d)), ov::Exception);
}

TEST(qsa_ops_gpu_test, paged_qsa_indexer_rejects_bad_rope_shapes) {
    split_dims d;
    using In = ov::intel_gpu::op::PagedQSAIndexer::InputIdx;
    const size_t spp = d.page_size / d.r;
    auto args = make_paged_indexer_args(d, 8, spp);
    args[In::ROPE_COS] = make_param({-1, I(d.rotary / 2)});  // a half-width row, not rotary
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::PagedQSAIndexer>(args, make_paged_indexer_config(d)), ov::Exception) << "rope_cos width";

    args = make_paged_indexer_args(d, 8, spp);
    args[In::ROPE_SIN] = make_param({3, I(d.rotary)});
    args[In::INDEX_QUERY] = make_param({4, I(d.Hiq * d.Di)});
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::PagedQSAIndexer>(args, make_paged_indexer_config(d)), ov::Exception) << "rope_sin tokens";

    args = make_paged_indexer_args(d, 8, spp);
    args[In::INDEX_BLOCK_ROPE_CACHE] = make_param({8, I(spp), I(d.rotary)});
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::PagedQSAIndexer>(args, make_paged_indexer_config(d)), ov::Exception) << "rope cache row width";

    args = make_paged_indexer_args(d, 8, spp);
    args[In::INDEX_BLOCK_ROPE_CACHE] = make_param({9, I(spp), I(2 * d.rotary)});
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::PagedQSAIndexer>(args, make_paged_indexer_config(d)), ov::Exception) << "rope cache num_blocks";

    args = make_paged_indexer_args(d, 8, spp);
    args[In::INDEX_BLOCK_ROPE_CACHE] = make_param({8, I(spp), I(2 * d.rotary)}, ov::element::f32);
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::PagedQSAIndexer>(args, make_paged_indexer_config(d)), ov::Exception) << "rope cache type";
}

TEST(qsa_ops_gpu_test, paged_qsa_indexer_requires_i32_integer_inputs) {
    split_dims d;
    using In = ov::intel_gpu::op::PagedQSAIndexer::InputIdx;
    for (size_t idx : {In::BLOCK_INDICES, In::BLOCK_INDICES_BEGINS, In::SUBSEQUENCE_BEGINS, In::PAST_LENS}) {
        auto args = make_paged_indexer_args(d, 8, d.page_size / d.r);
        args[idx] = make_param({-1}, ov::element::i64);
        EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::PagedQSAIndexer>(args, make_paged_indexer_config(d)), ov::Exception) << "input " << idx;
    }
}

TEST(qsa_ops_gpu_test, dense_qsa_indexer_validates_state_rope_and_key_length) {
    split_dims d;
    using In = ov::intel_gpu::op::QSAIndexer::InputIdx;
    using Out = ov::intel_gpu::op::QSAIndexer::OutputIdx;
    std::shared_ptr<ov::intel_gpu::op::QSAIndexer> node;
    ASSERT_NO_THROW(node = std::make_shared<ov::intel_gpu::op::QSAIndexer>(make_dense_indexer_args(d), make_dense_indexer_config(d)));
    // Outputs: selection plus the updated state, shaped and typed like the state inputs.
    ASSERT_EQ(node->get_output_size(), Out::COUNT);
    EXPECT_EQ(node->get_output_partial_shape(Out::SELECTED_BLOCK_INDICES), (ov::PartialShape{1, 1, -1, I(d.block_topk)}));
    for (auto [out, in] : {std::pair{Out::BLOCK_INDEX_KEYS_CACHE, In::BLOCK_INDEX_KEYS_CACHE},
                           std::pair{Out::PAST_INCOMPLETE_RAW_K, In::PAST_INCOMPLETE_RAW_K},
                           std::pair{Out::PAST_INCOMPLETE_ROPE, In::PAST_INCOMPLETE_ROPE}}) {
        EXPECT_EQ(node->get_output_partial_shape(out), node->get_input_partial_shape(in));
        EXPECT_EQ(node->get_output_element_type(out), ov::element::f16);
    }

    auto args = make_dense_indexer_args(d);
    args[In::ROPE_COS] = make_param({1, -1, I(d.rotary)}, ov::element::f32);
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::QSAIndexer>(args, make_dense_indexer_config(d)), ov::Exception) << "rope_cos type";

    args = make_dense_indexer_args(d);
    args[In::ROPE_COS] = make_param({1, -1, I(d.rotary / 2)});
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::QSAIndexer>(args, make_dense_indexer_config(d)), ov::Exception) << "half-width rope_cos";

    args = make_dense_indexer_args(d);
    args[In::ROPE_COS] = make_param({-1, I(d.rotary)});
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::QSAIndexer>(args, make_dense_indexer_config(d)), ov::Exception) << "rank-2 rope_cos";

    args = make_dense_indexer_args(d);
    args[In::ROPE_SIN] = make_param({2, -1, I(d.rotary)});
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::QSAIndexer>(args, make_dense_indexer_config(d)), ov::Exception) << "batch mismatch";

    args = make_dense_indexer_args(d);
    args[In::KEY_LENGTH] = make_param({1}, ov::element::i64);
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::QSAIndexer>(args, make_dense_indexer_config(d)), ov::Exception) << "i64 key_length";

    args = make_dense_indexer_args(d);
    args[In::PAST_INCOMPLETE_RAW_K] = make_param({1, 1, I(d.r), I(d.Di)});
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::QSAIndexer>(args, make_dense_indexer_config(d)), ov::Exception) << "r raw-key slots";

    args = make_dense_indexer_args(d);
    args[In::PAST_INCOMPLETE_ROPE] = make_param({1, I(d.rotary)});
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::QSAIndexer>(args, make_dense_indexer_config(d)), ov::Exception) << "non-interleaved rope state";

    args = make_dense_indexer_args(d);
    args[In::BLOCK_INDEX_KEYS_CACHE] = make_param({1, 1, -1, I(d.Di)}, ov::element::f32);
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::QSAIndexer>(args, make_dense_indexer_config(d)), ov::Exception) << "summary type";
}

cldnn::sparse_pa::Config make_sparse_pa_config(const split_dims& d) {
    cldnn::sparse_pa::Config c;
    c.num_heads = d.Hq;
    c.num_kv_heads = d.Hkv;
    c.num_selection_heads = 1;
    c.k_head_size = d.Dh;
    c.v_head_size = d.Dv;
    c.key_block_size = d.r;
    c.query_block_size = 1;
    c.causal = true;
    return c;
}

TEST(qsa_ops_gpu_test, sparse_pa_config_validation_rejects_illegal_attributes) {
    std::string reason;
    split_dims d;
    auto base = make_sparse_pa_config(d);
    ASSERT_TRUE(cldnn::sparse_pa::validate_config(base, reason)) << reason;

    auto c = base;
    c.num_kv_heads = 2;
    c.num_heads = 3;  // not a multiple of num_kv_heads
    ASSERT_FALSE(cldnn::sparse_pa::validate_config(c, reason));

    c = base;
    c.num_selection_heads = 3;  // not 1, num_kv_heads, or num_heads
    ASSERT_FALSE(cldnn::sparse_pa::validate_config(c, reason));

    c = base;
    c.query_block_size = 2;  // not implemented in this pass
    ASSERT_FALSE(cldnn::sparse_pa::validate_config(c, reason));
}

constexpr size_t paged_sparse_attention_scale_idx = 11;
constexpr size_t paged_sparse_attention_sinks_idx = 12;

ov::OutputVector make_sparse_pa_args(const split_dims& d, size_t num_pages) {
    return {make_param({-1, I(d.Hq * d.Dh)}),
            make_param({-1, I(d.Hkv * d.Dh)}),
            make_param({-1, I(d.Hkv * d.Dv)}),
            make_param({I(num_pages), I(d.Hkv), I(d.page_size), I(d.Dh)}),
            make_param({I(num_pages), I(d.Hkv), I(d.page_size), I(d.Dv)}),
            make_param({-1}, ov::element::i32),
            make_param({-1}, ov::element::i32),
            make_param({-1}, ov::element::i32),
            make_param({-1}, ov::element::i32),
            make_param({-1, 1, I(d.block_topk)}, ov::element::i32),
            make_param({-1, 1}, ov::element::i32),
            make_param({}),    // scale
            make_param({0})};  // sinks: empty = none
}

ov::intel_gpu::op::SparseSDPA::Config make_sparse_sdpa_config(const split_dims& d) {
    ov::intel_gpu::op::SparseSDPA::Config c;
    c.num_heads = d.Hq;
    c.num_kv_heads = d.Hkv;
    c.num_selection_heads = 1;
    c.k_head_size = d.Dh;
    c.v_head_size = d.Dv;
    c.key_block_size = d.r;
    c.query_block_size = 1;
    c.causal = true;
    return c;
}

TEST(qsa_ops_gpu_test, sparse_sdpa_config_validation_rejects_illegal_attributes) {
    std::string reason;
    split_dims d;
    auto base = make_sparse_sdpa_config(d);
    ASSERT_TRUE(ov::intel_gpu::op::SparseSDPA::validate_config(base, reason)) << reason;

    auto c = base;
    c.num_selection_heads = 3;
    ASSERT_FALSE(ov::intel_gpu::op::SparseSDPA::validate_config(c, reason));

    c = base;
    c.query_block_size = 2;
    ASSERT_FALSE(ov::intel_gpu::op::SparseSDPA::validate_config(c, reason));
}

ov::OutputVector make_sparse_sdpa_args(const split_dims& d, ov::element::Type_t et = ov::element::f16) {
    return {make_param({1, I(d.Hq), -1, I(d.Dh)}, et),
            make_param({1, I(d.Hkv), -1, I(d.Dh)}, et),
            make_param({1, I(d.Hkv), -1, I(d.Dv)}, et),
            make_param({1, 1, -1, I(d.block_topk)}, ov::element::i32),
            make_param({1, 1, -1}, ov::element::i32),
            make_param({1}, ov::element::i32)};
}

TEST(qsa_ops_gpu_test, sparse_sdpa_output_shape_and_optional_inputs) {
    split_dims d;
    auto args = make_sparse_sdpa_args(d);
    auto node = std::make_shared<ov::intel_gpu::op::SparseSDPA>(args, make_sparse_sdpa_config(d));
    ASSERT_FALSE(node->has_scale_input());
    ASSERT_FALSE(node->has_sinks());

    const auto& out_ps = node->get_output_partial_shape(0);
    ASSERT_EQ(out_ps.size(), 4u);
    ASSERT_EQ(out_ps[1].get_length(), I(d.Hq));
    ASSERT_EQ(out_ps[3].get_length(), I(d.Dv));

    args.push_back(make_param({}));  // scale
    auto node_with_scale = std::make_shared<ov::intel_gpu::op::SparseSDPA>(args, make_sparse_sdpa_config(d));
    ASSERT_TRUE(node_with_scale->has_scale_input());
    ASSERT_FALSE(node_with_scale->has_sinks());

    args.push_back(make_param({I(d.Hq)}));  // sinks
    auto node_with_sinks = std::make_shared<ov::intel_gpu::op::SparseSDPA>(args, make_sparse_sdpa_config(d));
    ASSERT_TRUE(node_with_sinks->has_sinks());

    auto clone = node->clone_with_new_inputs(node->input_values());
    ASSERT_NE(std::dynamic_pointer_cast<ov::intel_gpu::op::SparseSDPA>(clone), nullptr);

    args = make_sparse_sdpa_args(d);
    args[SparseSDPAIn::SELECTED_BLOCK_INDICES] = make_param({1, 1, -1, I(d.block_topk)}, ov::element::i64);
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::SparseSDPA>(args, make_sparse_sdpa_config(d)), ov::Exception);

    args = make_sparse_sdpa_args(d);
    args[SparseSDPAIn::KEY_LENGTH] = make_param({1}, ov::element::i64);
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::SparseSDPA>(args, make_sparse_sdpa_config(d)), ov::Exception) << "i64 key_length";

    args = make_sparse_sdpa_args(d);
    args[SparseSDPAIn::KEY_LENGTH] = make_param({2}, ov::element::i32);
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::SparseSDPA>(args, make_sparse_sdpa_config(d)), ov::Exception) << "key_length batch";

    args = make_sparse_sdpa_args(d);
    args.pop_back();
    EXPECT_THROW((void)std::make_shared<ov::intel_gpu::op::SparseSDPA>(args, make_sparse_sdpa_config(d)), ov::Exception) << "key_length is required";
}

// ---------------------------------------------------------------------------
// Plugin-level rejection of configurations outside the v1 GPU scope
// (H_ik == 1, num_selection_heads == 1, f16 data). Goes through ov::Core so the ProgramBuilder
// factories are exercised; no device support is needed since lowering fails first.
// ---------------------------------------------------------------------------

bool gpu_in_core(ov::Core& core) {
    std::vector<std::string> devices;
    try {
        devices = core.get_available_devices();
    } catch (...) {
    }
    return std::any_of(devices.begin(), devices.end(), [](const std::string& dev) {
        return dev.rfind("GPU", 0) == 0;
    });
}

void expect_compile_error(ov::Core& core, const std::shared_ptr<ov::Node>& op, const ov::AnyMap& props, const std::string& needle) {
    ov::ParameterVector params;
    for (const auto& input : op->input_values()) {
        if (auto p = ov::as_type_ptr<ov::op::v0::Parameter>(input.get_node_shared_ptr())) {
            params.push_back(p);
        }
    }
    ov::ResultVector results;
    for (const auto& output : op->outputs()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(output));
    }
    auto model = std::make_shared<ov::Model>(results, params);
    try {
        core.compile_model(model, "GPU", props);
        ADD_FAILURE() << "compile_model unexpectedly succeeded; expected an error containing '" << needle << "'";
    } catch (const ov::Exception& e) {
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
    }
}

TEST(qsa_ops_gpu_test, plugin_rejects_unsupported_indexer_heads_and_precision) {
    ov::Core core;
    if (!gpu_in_core(core)) {
        GTEST_SKIP() << "GPU plugin is not available in ov::Core";
    }
    split_dims d;
    d.Hiq = 2;

    auto make_dense = [&](size_t h_ik, size_t num_selection_heads) {
        auto args = make_dense_indexer_args(d, h_ik);
        args[DenseIdxIn::Q_NORM_WEIGHT] = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{d.Di}, std::vector<float>(d.Di, 1.0f));
        args[DenseIdxIn::K_NORM_WEIGHT] = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{d.Di}, std::vector<float>(d.Di, 1.0f));
        auto cfg = make_dense_indexer_config(d);
        cfg.num_selection_heads = num_selection_heads;
        return std::make_shared<ov::intel_gpu::op::QSAIndexer>(args, cfg);
    };
    expect_compile_error(core, make_dense(2, 1), {}, "only H_ik == 1 is supported");
    expect_compile_error(core, make_dense(2, 2), {}, "only num_selection_heads == 1 is supported");

    auto sdpa = std::make_shared<ov::intel_gpu::op::SparseSDPA>(make_sparse_sdpa_args(d, ov::element::f32), make_sparse_sdpa_config(d));
    expect_compile_error(core, sdpa, {ov::hint::inference_precision(ov::element::f32)}, "must be f16");

    auto sink_args = make_sparse_sdpa_args(d);
    sink_args.push_back(ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.25f}));
    sink_args.push_back(make_param({I(d.Hq)}));
    auto sdpa_sinks = std::make_shared<ov::intel_gpu::op::SparseSDPA>(sink_args, make_sparse_sdpa_config(d));
    expect_compile_error(core, sdpa_sinks, {}, "sinks input is not supported");
}

TEST(qsa_ops_gpu_test, plugin_lowers_sparse_pa_scale_input_and_rejects_sinks) {
    ov::Core core;
    if (!gpu_in_core(core)) {
        GTEST_SKIP() << "GPU plugin is not available in ov::Core";
    }
    split_dims d;
    const auto key_block_size = static_cast<int64_t>(d.r);

    // A non-empty sinks input is rejected, a non-Constant scale input is rejected.
    auto args = make_sparse_pa_args(d, 8);
    args[paged_sparse_attention_sinks_idx] = make_param({I(d.Hq)});
    expect_compile_error(core, std::make_shared<ov::op::internal::PagedSparseAttention>(args, key_block_size), {}, "sinks input is not supported");

    args = make_sparse_pa_args(d, 8);
    args[paged_sparse_attention_sinks_idx] = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{0}, std::vector<float>{});
    expect_compile_error(core, std::make_shared<ov::op::internal::PagedSparseAttention>(args, key_block_size), {}, "scale input must be a Constant");

    // Constant scale + empty sinks lowers (scale and sinks are not primitive inputs).
    if (!qsa_ops_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    args[paged_sparse_attention_scale_idx] = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.0625f});
    auto op = std::make_shared<ov::op::internal::PagedSparseAttention>(args, key_block_size);
    ov::ParameterVector params;
    for (const auto& input : op->input_values()) {
        if (auto p = ov::as_type_ptr<ov::op::v0::Parameter>(input.get_node_shared_ptr())) {
            params.push_back(p);
        }
    }
    auto model = std::make_shared<ov::Model>(ov::OutputVector{op->output(0)}, params);
    ASSERT_NO_THROW(core.compile_model(model, "GPU", ov::hint::inference_precision(ov::element::f16)));
}

}  // namespace

// ---------------------------------------------------------------------------
// Performance (DISABLED_ by default). Wall-clock per synchronous execute(), which
// includes host dispatch and plan uploads, with Qwen3.8-Flash-Next shapes. The traffic
// estimate assumes no cache reuse across query tokens, so prefill GB/s is an upper bound.
// ---------------------------------------------------------------------------

namespace {

struct perf_case {
    bool paged;
    size_t past_len;
    size_t q_len;
    bool no_dpas_score = false;  // Di=16 fails the Q2 dpas gate, forcing the non-dpas Q2 path
};

struct perf_timing {
    double avg_ms = 0.0;
    double min_ms = 0.0;
};

perf_timing time_network(network& net, const primitive_id& out_id, int warmup, int iterations) {
    auto run_once = [&]() {
        auto outputs = net.execute();
        outputs.at(out_id).get_memory();
        net.get_stream().finish();
    };
    for (int i = 0; i < warmup; i++) {
        run_once();
    }
    perf_timing t;
    t.min_ms = std::numeric_limits<double>::max();
    double total = 0.0;
    for (int i = 0; i < iterations; i++) {
        const auto start = std::chrono::steady_clock::now();
        run_once();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        total += ms;
        t.min_ms = std::min(t.min_ms, ms);
    }
    t.avg_ms = total / iterations;
    return t;
}

double indexer_bytes(const split_dims& d, size_t past_len, size_t q_len) {
    const double n_blocks = static_cast<double>((past_len + q_len) / d.r);
    const double tiles = static_cast<double>((q_len + 15) / 16);
    return q_len * (d.Hiq + 1) * d.Di * 2.0 + tiles * n_blocks * d.Di * 2.0 + q_len * n_blocks * 4.0 * 2.0 + q_len * d.block_topk * 4.0;
}

double attention_bytes(const split_dims& d, size_t past_len, size_t q_len) {
    double bytes = q_len * d.Hq * (d.Dh + d.Dv) * 2.0;
    for (size_t t = 0; t < q_len; t++) {
        const size_t n_complete = (past_len + t + 1) / d.r;
        const double tokens = static_cast<double>(std::min(d.block_topk, n_complete) * d.r + d.r);
        bytes += tokens * d.Hkv * (d.Dh + d.Dv) * 2.0;
    }
    return bytes;
}

void print_perf(const char* op, const perf_case& c, const perf_timing& t, double bytes) {
    const double gbps = bytes / (t.avg_ms * 1e-3) / 1e9;
    std::cout << "QSA_PERF " << std::left << std::setw(16) << op << " mode=" << std::setw(7) << (c.q_len == 1 ? "decode" : "prefill") << " ctx=" << std::setw(6)
              << (c.past_len + c.q_len) << " q=" << std::setw(5) << c.q_len << std::right << std::fixed << std::setprecision(3) << " avg_ms=" << std::setw(8)
              << t.avg_ms << " min_ms=" << std::setw(8) << t.min_ms << std::setprecision(1) << " est_MB=" << std::setw(7) << bytes / 1e6
              << " eff_GB/s=" << std::setw(6) << gbps << " (BMG peak 456)" << std::endl;
}

class qsa_ops_perf : public ::testing::TestWithParam<perf_case> {};

TEST_P(qsa_ops_perf, qwen38) {
    if (!qsa_ops_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    const auto c = GetParam();
    auto d = qwen38_dims();
    if (c.no_dpas_score) {
        d.Di = 16;
        d.rotary = 8;
    }
    auto& engine = get_test_engine();
    // Microsecond kernels never ramp the GPU clock in 3 warmups: QSA_PERF_WARMUP / QSA_PERF_ITERS settle it.
    auto env_int = [](const char* name, int fallback) {
        const char* v = std::getenv(name);
        return v ? static_cast<int>(std::strtol(v, nullptr, 10)) : fallback;
    };
    const int warmup = env_int("QSA_PERF_WARMUP", 3);
    const int iterations = env_int("QSA_PERF_ITERS", 20);

    const char* indexer_name = c.paged ? "PagedQSAIndexer" : "QSAIndexer";
    const char* attention_name = c.paged ? "SparsePA" : "SparseSDPA";
    const double idx_bytes = indexer_bytes(d, c.past_len, c.q_len);
    const double attn_bytes = attention_bytes(d, c.past_len, c.q_len);

    std::unique_ptr<network> idx_net, attn_net, pair_net;
    if (c.paged) {
        const auto in = paged_inputs_synthetic(d, c.past_len, c.q_len);
        idx_net = build_paged_network(engine, in, Stage::IndexerOnly);
        attn_net = build_paged_network(engine, in, Stage::AttentionOnly);
        pair_net = build_paged_network(engine, in, Stage::Pair);
    } else {
        const auto in = dense_inputs_synthetic(d, c.past_len, c.q_len);
        idx_net = build_dense_network(engine, in, Stage::IndexerOnly);
        attn_net = build_dense_network(engine, in, Stage::AttentionOnly);
        pair_net = build_dense_network(engine, in, Stage::Pair);
    }
    print_perf(indexer_name, c, time_network(*idx_net, "indexer", warmup, iterations), idx_bytes);
    print_perf(attention_name, c, time_network(*attn_net, "attn", warmup, iterations), attn_bytes);
    print_perf(c.paged ? "Paged pair" : "Dense pair", c, time_network(*pair_net, "attn", warmup, iterations), idx_bytes + attn_bytes);
}

INSTANTIATE_TEST_SUITE_P(DISABLED_qsa_ops_perf,
                         qsa_ops_perf,
                         ::testing::Values(perf_case{true, 511, 1},
                                           perf_case{true, 1023, 1},
                                           perf_case{true, 2047, 1},
                                           perf_case{true, 4095, 1},
                                           perf_case{true, 16383, 1},
                                           perf_case{true, 32767, 1},
                                           perf_case{true, 3072, 1024},
                                           perf_case{true, 0, 4096},
                                           perf_case{true, 3072, 1024, true},
                                           perf_case{true, 0, 4096, true},
                                           perf_case{false, 511, 1},
                                           perf_case{false, 1023, 1},
                                           perf_case{false, 2047, 1},
                                           perf_case{false, 4095, 1},
                                           perf_case{false, 16383, 1},
                                           perf_case{false, 32767, 1},
                                           perf_case{false, 3072, 1024},
                                           perf_case{false, 0, 4096},
                                           perf_case{false, 3072, 1024, true},
                                           perf_case{false, 0, 4096, true}),
                         [](const ::testing::TestParamInfo<perf_case>& info) {
                             const auto& c = info.param;
                             return std::string(c.paged ? "paged" : "dense") + "_ctx" + std::to_string(c.past_len + c.q_len) + "_q" + std::to_string(c.q_len) +
                                    (c.no_dpas_score ? "_nodpasq2" : "");
                         });

// Per-kernel sweep driver for external GPU tracing (e.g. onetrace -d): one dense (QSAIndexer +
// SparseSDPA) or paged (PagedQSAIndexer + SparsePA) pair with Qwen3.8 dims, configured via the
// environment: QSA_PERF_IMPL=dense|paged, QSA_PERF_PHASE=prefill|decode, QSA_PERF_B, QSA_PERF_LEN
// (prompt length / history length), QSA_PERF_ITERS, QSA_PERF_WARMUP, QSA_PERF_FLUSH_MB (writes
// that many MiB between iterations to evict L2), QSA_PERF_STAGES (comma list of pair|indexer|attention, default pair).
TEST(qsa_ops_perf_sweep, DISABLED_pair) {
    if (!qsa_ops_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    auto env_size = [](const char* name, size_t fallback) {
        const char* v = std::getenv(name);
        return v ? static_cast<size_t>(std::strtoull(v, nullptr, 10)) : fallback;
    };
    const char* phase_env = std::getenv("QSA_PERF_PHASE");
    const std::string phase = phase_env ? phase_env : "prefill";
    ASSERT_TRUE(phase == "prefill" || phase == "decode") << phase;
    const size_t batch = env_size("QSA_PERF_B", 1);
    const size_t len = env_size("QSA_PERF_LEN", 1024);
    const int iterations = static_cast<int>(env_size("QSA_PERF_ITERS", 30));
    const size_t flush_mb = env_size("QSA_PERF_FLUSH_MB", 0);
    const int warmup = static_cast<int>(env_size("QSA_PERF_WARMUP", 5));

    auto& engine = get_test_engine();
    const char* impl_env = std::getenv("QSA_PERF_IMPL");
    const std::string impl = impl_env ? impl_env : "dense";
    ASSERT_TRUE(impl == "dense" || impl == "paged") << impl;
    const size_t past = phase == "prefill" ? 0 : len;
    const size_t q_len = phase == "prefill" ? len : 1;
    const char* stages_env = std::getenv("QSA_PERF_STAGES");
    std::stringstream stages(stages_env ? stages_env : "pair");
    memory::ptr flush;
    if (flush_mb != 0) {
        flush = engine.allocate_memory(layout{ov::PartialShape{I(flush_mb << 20)}, data_types::u8, format::bfyx}, allocation_type::usm_device);
    }
    std::string stage_name;
    while (std::getline(stages, stage_name, ',')) {
        ASSERT_TRUE(stage_name == "pair" || stage_name == "indexer" || stage_name == "attention") << stage_name;
        const Stage stage = stage_name == "pair" ? Stage::Pair : (stage_name == "indexer" ? Stage::IndexerOnly : Stage::AttentionOnly);
        const std::string out_id = stage == Stage::IndexerOnly ? "indexer" : "attn";
        std::unique_ptr<network> net;
        if (impl == "paged") {
            net = build_paged_network(engine, paged_inputs_synthetic(qwen38_dims(), past, q_len, batch), stage);
        } else {
            net = build_dense_network(engine, dense_inputs_synthetic(qwen38_dims(), past, q_len, batch), stage);
        }
        auto run_once = [&]() {
            if (flush) {
                flush->fill(get_test_stream(), static_cast<unsigned char>(0), {}, true);
            }
            const auto start = std::chrono::steady_clock::now();
            auto outputs = net->execute();
            outputs.at(out_id).get_memory();
            net->get_stream().finish();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        };
        for (int i = 0; i < warmup; i++) {
            run_once();
        }
        double total = 0.0;
        double best = std::numeric_limits<double>::max();
        for (int i = 0; i < iterations; i++) {
            const double ms = run_once();
            total += ms;
            best = std::min(best, ms);
        }
        std::cout << "QSA_SWEEP impl=" << impl << " phase=" << phase << " B=" << batch << " len=" << len << " calls=" << warmup + iterations
                  << " wall_avg_ms=" << total / iterations << " wall_min_ms=" << best << " flush_mb=" << flush_mb << " stage=" << stage_name << std::endl;
    }
}

}  // namespace
