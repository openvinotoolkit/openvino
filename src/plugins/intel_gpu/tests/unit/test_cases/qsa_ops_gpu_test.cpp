// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the split PagedQSAIndexer/SparsePA and QSAIndexer/SparseSDPA
// primitives (qsa_final_design_implement.md). These primitives are ADDITIVE
// infrastructure in this pass -- there is no automatic graph-rewrite pass yet that
// produces them from a generic model graph (see the design doc's Phase 4 gate), so
// topologies here are built directly at the cldnn primitive level, mirroring
// qsa_gpu_test.cpp's own raw-topology style for the original fused qsa primitive.
//
// Covered:
//  - PagedQSAIndexer: complete-block-only selection (count excludes the causal tail),
//    raw-K stored unnormalized/unrotated, summary mean-pool + RMSNorm + block-start RoPE.
//  - SparsePA: Q0 main K/V cache write + Q3 selected-block attention with the
//    automatically-derived causal tail, no output gate (v1 scope).
//  - QSAIndexer / SparseSDPA: same math via direct dense (non-paged) addressing, no
//    page table, bottom-right-aligned causal position q_abs = S - L + l.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/paged_qsa_indexer.hpp>
#include <intel_gpu/primitives/qsa_indexer.hpp>
#include <intel_gpu/primitives/sparse_pa.hpp>
#include <intel_gpu/primitives/sparse_sdpa.hpp>
#include <openvino/op/parameter.hpp>

#include "impls/ocl/kernel_selector_helper.h"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

using PagedIn = ov::intel_gpu::op::PagedQSAIndexer::InputIdx;
using DenseIdxIn = ov::intel_gpu::op::QSAIndexer::InputIdx;
using SparsePAIn = ov::intel_gpu::op::SparsePA::InputIdx;
using SparseSDPAIn = ov::intel_gpu::op::SparseSDPA::InputIdx;

bool cm_qsa_ops_available() {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    return cldnn::check_cm_jit_support(engine, config) && engine.get_device_info().supports_immad;
}

struct split_dims {
    size_t Hiq = 2;   // index-query heads (H_ik == 1 fixed in v1)
    size_t Di = 16;   // indexer head dim
    size_t rotary = 8;
    size_t r = 4;      // compress_ratio
    size_t block_topk = 2;
    size_t page_size = 16;
    size_t max_positions = 64;
    float eps = 1e-6f;

    size_t Hq = 2;
    size_t Hkv = 1;
    size_t Dh = 16;
    size_t Dv = 16;
};

#define ASSERT_SELECTION_IS_DISCRIMINATING(d, past_len, q_len)                                                             \
    do {                                                                                                                   \
        const size_t n_complete_ = ((past_len) + (q_len)) / (d).r;                                                         \
        ASSERT_GE(n_complete_, 3 * (d).block_topk) << "the top-k selects " << (d).block_topk << " of only " << n_complete_ \
                                                   << " blocks, so a wrong selection cannot change the output";            \
    } while (0)

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

std::vector<float> random_vec(tests::random_generator& rg, size_t count, float scale, float bias = 0.0f) {
    auto v = rg.generate_random_1d<float>(count, -1, 1, 16);
    std::vector<float> out(v.begin(), v.end());
    for (auto& x : out) {
        x = x * scale + bias;
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

std::vector<float> rotate_half(const std::vector<float>& x) {
    const size_t half = x.size() / 2;
    std::vector<float> out(x.size());
    for (size_t j = 0; j < half; j++) {
        out[j] = -x[half + j];
        out[half + j] = x[j];
    }
    return out;
}

void apply_rope(std::vector<float>& v, const std::vector<float>& cos_full, const std::vector<float>& sin_full, size_t rotary, size_t position) {
    const std::vector<float> x_rope(v.begin(), v.begin() + rotary);
    const auto rotated = rotate_half(x_rope);
    for (size_t j = 0; j < rotary; j++) {
        v[j] = x_rope[j] * cos_full[position * rotary + j] + rotated[j] * sin_full[position * rotary + j];
    }
}

/// CPU reference for the split Indexer + Attention pair (H_ik == 1, H_sel == 1,
/// q_block_size == 1, no output gate). `q_idx`/`k_idx` are already "projected" (the new
/// ops never project), so this reference starts directly from those raw vectors.
struct split_reference {
    split_dims d;
    size_t past_len = 0;
    size_t q_len = 0;

    std::vector<float> k_idx_all;  // [total_len, Di]
    std::vector<float> q_idx_cur;  // [q_len, Hiq, Di]
    std::vector<float> key_all;    // [total_len, Hkv*Dh]
    std::vector<float> value_all;  // [total_len, Hkv*Dv]
    std::vector<float> query_cur;  // [q_len, Hq*Dh]
    std::vector<float> q_norm, k_norm;
    std::vector<float> rope_cos, rope_sin, rotary;

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
                pooled[k] += k_idx_all[p * d.Di + k];
            }
        }
        for (float& x : pooled) {
            x /= static_cast<float>(d.r);
        }
        rms_norm(pooled, k_norm, d.eps);
        apply_rope(pooled, rope_cos, rope_sin, d.rotary, block_id * d.r);
        summary_memo.emplace(block_id, pooled);
        return pooled;
    }

    std::vector<std::pair<float, size_t>> scored_blocks(size_t t) const {
        const size_t abs_pos = past_len + t;
        const size_t n_complete = (abs_pos + 1) / d.r;

        std::vector<std::vector<float>> heads(d.Hiq);
        for (size_t h = 0; h < d.Hiq; h++) {
            heads[h].assign(q_idx_cur.begin() + (t * d.Hiq + h) * d.Di, q_idx_cur.begin() + (t * d.Hiq + h + 1) * d.Di);
            rms_norm(heads[h], q_norm, d.eps);
            apply_rope(heads[h], rope_cos, rope_sin, d.rotary, abs_pos);
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

    float margin_of(const std::vector<std::pair<float, size_t>>& scored) const {
        if (scored.size() <= d.block_topk) {
            return std::numeric_limits<float>::infinity();
        }
        const float lo = scored[d.block_topk - 1].first;
        const float hi = scored[d.block_topk].first;
        return (lo - hi) / std::max(std::fabs(lo), 1e-3f);
    }

    std::vector<float> attention(std::vector<float>* margins_out = nullptr) const {
        std::vector<float> out(q_len * d.Hq * d.Dv, 0.0f);
        if (margins_out != nullptr) {
            margins_out->assign(q_len, 0.0f);
        }
        for (size_t t = 0; t < q_len; t++) {
            const size_t abs_pos = past_len + t;
            const auto scored = scored_blocks(t);
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
                positions.push_back(p);
            }

            for (size_t h = 0; h < d.Hq; h++) {
                const size_t kv_head = h / (d.Hq / d.Hkv);
                const size_t q_base = t * d.Hq * d.Dh + h * d.Dh;

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
                    out[(t * d.Hq + h) * d.Dv + j] = static_cast<float>(acc / sum);
                }
            }
        }
        return out;
    }
};

split_reference make_split_reference(tests::random_generator& rg, const split_dims& d, size_t past_len, size_t q_len) {
    split_reference ref;
    ref.d = d;
    ref.past_len = past_len;
    ref.q_len = q_len;
    const size_t total = past_len + q_len;

    ref.k_idx_all = random_vec(rg, total * d.Di, 1.0f);
    ref.q_idx_cur = random_vec(rg, q_len * d.Hiq * d.Di, 1.0f);
    ref.key_all = random_vec(rg, total * d.Hkv * d.Dh, 1.0f);
    ref.value_all = random_vec(rg, total * d.Hkv * d.Dv, 1.0f);
    ref.query_cur = random_vec(rg, q_len * d.Hq * d.Dh, 1.0f);
    ref.q_norm = random_vec(rg, d.Di, 1.0f, 1.0f);
    ref.k_norm = random_vec(rg, d.Di, 1.0f, 1.0f);

    ref.rope_cos.assign(d.max_positions * d.rotary, 1.0f);
    ref.rope_sin.assign(d.max_positions * d.rotary, 0.0f);
    ref.rotary.assign(d.max_positions * d.rotary, 0.0f);
    for (size_t p = 0; p < d.max_positions; p++) {
        for (size_t j = 0; j < d.rotary / 2; j++) {
            const float freq = 1.0f / std::pow(10000.0f, static_cast<float>(2 * j) / static_cast<float>(d.rotary));
            const float angle = static_cast<float>(p) * freq;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            ref.rope_cos[p * d.rotary + j] = c;
            ref.rope_cos[p * d.rotary + d.rotary / 2 + j] = c;
            ref.rope_sin[p * d.rotary + j] = s;
            ref.rope_sin[p * d.rotary + d.rotary / 2 + j] = s;
            ref.rotary[p * d.rotary + 2 * j] = c;
            ref.rotary[p * d.rotary + 2 * j + 1] = s;
        }
    }
    return ref;
}

TEST(qsa_ops_gpu_test, paged_pair_matches_reference) {
    if (!cm_qsa_ops_available()) {
        GTEST_SKIP() << "CM is not supported on this device";
    }
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    split_dims d;
    const size_t past_len = 20;
    const size_t q_len = 44;  // total 64 -> 16 complete blocks, well above 3*block_topk
    ASSERT_SELECTION_IS_DISCRIMINATING(d, past_len, q_len);

    auto ref = make_split_reference(rg, d, past_len, q_len);
    const size_t total = ref.total_len();
    const size_t num_pages = (total + d.page_size - 1) / d.page_size;
    const size_t summaries_per_page = d.page_size / d.r;

    std::vector<float> key_cache(num_pages * d.Hkv * d.page_size * d.Dh, 0.0f);
    std::vector<float> value_cache(num_pages * d.Hkv * d.page_size * d.Dv, 0.0f);
    std::vector<float> raw_k_cache(num_pages * d.page_size * d.Di, 0.0f);
    std::vector<float> summary_cache(num_pages * summaries_per_page * d.Di, 0.0f);
    std::vector<int32_t> position_cache(num_pages * summaries_per_page, 0);

    for (size_t p = 0; p < past_len; p++) {
        const size_t page = p / d.page_size;
        const size_t slot = p % d.page_size;
        for (size_t h = 0; h < d.Hkv; h++) {
            for (size_t j = 0; j < d.Dh; j++) {
                key_cache[((page * d.Hkv + h) * d.page_size + slot) * d.Dh + j] = ref.key_all[p * d.Hkv * d.Dh + h * d.Dh + j];
            }
            for (size_t j = 0; j < d.Dv; j++) {
                value_cache[((page * d.Hkv + h) * d.page_size + slot) * d.Dv + j] = ref.value_all[p * d.Hkv * d.Dv + h * d.Dv + j];
            }
        }
        for (size_t j = 0; j < d.Di; j++) {
            raw_k_cache[(page * d.page_size + slot) * d.Di + j] = ref.k_idx_all[p * d.Di + j];
        }
    }
    for (size_t b = 0; b * d.r + d.r <= past_len; b++) {
        const auto s = ref.summary(b);
        const size_t page = b / summaries_per_page;
        const size_t slot = b % summaries_per_page;
        for (size_t j = 0; j < d.Di; j++) {
            summary_cache[(page * summaries_per_page + slot) * d.Di + j] = s[j];
        }
        // This test's position_ids convention is the trivial sequential one
        // (position == absolute token index), matching the reference's own
        // `apply_rope(..., block_id * d.r)`/`apply_rope(..., abs_pos)` calls. One
        // recorded position per block (its first token), same granularity as summary_cache.
        position_cache[page * summaries_per_page + slot] = static_cast<int32_t>(b * d.r);
    }

    std::vector<int32_t> block_indices(num_pages);
    std::iota(block_indices.begin(), block_indices.end(), 0);
    const std::vector<int32_t> block_indices_begins = {0, static_cast<int32_t>(num_pages)};
    const std::vector<int32_t> past_lens = {static_cast<int32_t>(past_len)};
    const std::vector<int32_t> subsequence_begins = {0, static_cast<int32_t>(q_len)};
    std::vector<int32_t> position_ids(q_len);
    for (size_t t = 0; t < q_len; t++) {
        position_ids[t] = static_cast<int32_t>(past_len + t);
    }
    const std::vector<int32_t> qb_start = {0, static_cast<int32_t>(q_len)};

    const auto cur_key = std::vector<float>(ref.key_all.begin() + past_len * d.Hkv * d.Dh, ref.key_all.end());
    const auto cur_value = std::vector<float>(ref.value_all.begin() + past_len * d.Hkv * d.Dv, ref.value_all.end());
    const auto cur_k_idx = std::vector<float>(ref.k_idx_all.begin() + past_len * d.Di, ref.k_idx_all.end());

    ov::intel_gpu::op::PagedQSAIndexer::Config idx_cfg;
    idx_cfg.compress_ratio = d.r;
    idx_cfg.block_topk = d.block_topk;
    idx_cfg.h_sel = 1;
    idx_cfg.indexer_rotary_dim = d.rotary;
    idx_cfg.indexer_head_dim = d.Di;
    idx_cfg.page_size = d.page_size;
    idx_cfg.eps = d.eps;
    idx_cfg.scale = 1.0f / std::sqrt(static_cast<float>(d.Di));

    ov::intel_gpu::op::SparsePA::Config attn_cfg;
    attn_cfg.num_heads = d.Hq;
    attn_cfg.num_kv_heads = d.Hkv;
    attn_cfg.h_sel = 1;
    attn_cfg.k_head_size = d.Dh;
    attn_cfg.v_head_size = d.Dv;
    attn_cfg.block_size = d.page_size;
    attn_cfg.sel_block_size = d.r;
    attn_cfg.q_block_size = 1;
    attn_cfg.scale = 1.0f / std::sqrt(static_cast<float>(d.Dh));
    attn_cfg.causal = true;

    topology topo;
    topo.add(input_layout("q_idx", layout{{static_cast<int64_t>(q_len), static_cast<int64_t>(d.Hiq * d.Di)}, data_types::f16, format::bfyx}));
    topo.add(input_layout("k_idx", layout{{static_cast<int64_t>(q_len), static_cast<int64_t>(d.Di)}, data_types::f16, format::bfyx}));
    topo.add(input_layout("position_ids", layout{{static_cast<int64_t>(q_len)}, data_types::i32, format::bfyx}));
    topo.add(data("rotary", make_f16(engine, {static_cast<int64_t>(d.max_positions), static_cast<int64_t>(d.rotary)}, ref.rotary)));
    topo.add(input_layout(
        "summary_cache",
        layout{{static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)}, data_types::f16, format::bfyx}));
    topo.add(input_layout(
        "raw_k_cache", layout{{static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(d.page_size), static_cast<int64_t>(d.Di)}, data_types::f16, format::bfyx}));
    topo.add(data("q_norm", make_f16(engine, {static_cast<int64_t>(d.Di)}, ref.q_norm)));
    topo.add(data("k_norm", make_f16(engine, {static_cast<int64_t>(d.Di)}, ref.k_norm)));
    topo.add(data("block_indices", make_i32(engine, {static_cast<int64_t>(num_pages)}, block_indices)));
    topo.add(data("block_indices_begins", make_i32(engine, {2}, block_indices_begins)));
    topo.add(data("subsequence_begins", make_i32(engine, {2}, subsequence_begins)));
    topo.add(data("past_lens", make_i32(engine, {1}, past_lens)));
    topo.add(input_layout("position_cache",
                          layout{{static_cast<int64_t>(num_pages), static_cast<int64_t>(summaries_per_page)}, data_types::i32, format::bfyx}));

    const std::vector<input_info> idx_inputs = {input_info("q_idx"),
                                                input_info("k_idx"),
                                                input_info("position_ids"),
                                                input_info("rotary"),
                                                input_info("summary_cache"),
                                                input_info("raw_k_cache"),
                                                input_info("q_norm"),
                                                input_info("k_norm"),
                                                input_info("block_indices"),
                                                input_info("block_indices_begins"),
                                                input_info("subsequence_begins"),
                                                input_info("past_lens"),
                                                input_info("position_cache")};
    static_assert(PagedIn::COUNT == 13, "keep the test input order in sync with PagedQSAIndexer::InputIdx");
    topo.add(paged_qsa_indexer("indexer", idx_inputs, idx_cfg));

    topo.add(input_layout("query", layout{{static_cast<int64_t>(q_len), static_cast<int64_t>(d.Hq * d.Dh)}, data_types::f16, format::bfyx}));
    topo.add(input_layout("key", layout{{static_cast<int64_t>(q_len), static_cast<int64_t>(d.Hkv * d.Dh)}, data_types::f16, format::bfyx}));
    topo.add(input_layout("value", layout{{static_cast<int64_t>(q_len), static_cast<int64_t>(d.Hkv * d.Dv)}, data_types::f16, format::bfyx}));
    topo.add(input_layout(
        "key_cache",
        layout{{static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.page_size), static_cast<int64_t>(d.Dh)},
               data_types::f16,
               format::bfyx}));
    topo.add(input_layout(
        "value_cache",
        layout{{static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.page_size), static_cast<int64_t>(d.Dv)},
               data_types::f16,
               format::bfyx}));
    topo.add(data("qb_start", make_i32(engine, {2}, qb_start)));

    const std::vector<input_info> attn_inputs = {input_info("query"),
                                                 input_info("key"),
                                                 input_info("value"),
                                                 input_info("key_cache"),
                                                 input_info("value_cache"),
                                                 input_info("block_indices"),
                                                 input_info("block_indices_begins"),
                                                 input_info("subsequence_begins"),
                                                 input_info("past_lens"),
                                                 input_info("indexer", 0),
                                                 input_info("indexer", 1),
                                                 input_info("qb_start")};
    static_assert(SparsePAIn::COUNT == 12, "keep the test input order in sync with SparsePA::InputIdx");
    topo.add(sparse_pa("attn", attn_inputs, attn_cfg));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::optimize_data(true));
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topo, cfg);

    network.set_input_data("q_idx", make_f16(engine, {static_cast<int64_t>(q_len), static_cast<int64_t>(d.Hiq * d.Di)}, ref.q_idx_cur));
    network.set_input_data("k_idx", make_f16(engine, {static_cast<int64_t>(q_len), static_cast<int64_t>(d.Di)}, cur_k_idx));
    network.set_input_data("position_ids", make_i32(engine, {static_cast<int64_t>(q_len)}, position_ids));
    network.set_input_data(
        "summary_cache",
        make_f16(engine, {static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)}, summary_cache));
    network.set_input_data(
        "raw_k_cache", make_f16(engine, {static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(d.page_size), static_cast<int64_t>(d.Di)}, raw_k_cache));
    network.set_input_data("position_cache", make_i32(engine, {static_cast<int64_t>(num_pages), static_cast<int64_t>(summaries_per_page)}, position_cache));
    network.set_input_data("query", make_f16(engine, {static_cast<int64_t>(q_len), static_cast<int64_t>(d.Hq * d.Dh)}, ref.query_cur));
    network.set_input_data("key", make_f16(engine, {static_cast<int64_t>(q_len), static_cast<int64_t>(d.Hkv * d.Dh)}, cur_key));
    network.set_input_data("value", make_f16(engine, {static_cast<int64_t>(q_len), static_cast<int64_t>(d.Hkv * d.Dv)}, cur_value));
    network.set_input_data(
        "key_cache",
        make_f16(engine,
                 {static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.page_size), static_cast<int64_t>(d.Dh)},
                 key_cache));
    network.set_input_data(
        "value_cache",
        make_f16(engine,
                 {static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.page_size), static_cast<int64_t>(d.Dv)},
                 value_cache));

    auto outputs = network.execute();
    cldnn::mem_lock<ov::float16> out(outputs.at("attn").get_memory(), get_test_stream());

    std::vector<float> margins;
    const auto expected = ref.attention(&margins);
    ASSERT_EQ(out.size(), expected.size());

    size_t compared = 0;
    for (size_t t = 0; t < q_len; t++) {
        if (!std::isinf(margins[t]) && margins[t] < 0.02f) {
            continue;  // near-tied top-k boundary, see intel_gpu_cm_kernels.md notes
        }
        compared++;
        for (size_t i = 0; i < d.Hq * d.Dv; i++) {
            const size_t idx = t * d.Hq * d.Dv + i;
            EXPECT_NEAR(static_cast<float>(out[idx]), expected[idx], 2e-2f) << "t=" << t << " i=" << i;
        }
    }
    ASSERT_GE(compared, q_len / 2) << "too many tokens were skipped as near-tied; test is not discriminating";
}

TEST(qsa_ops_gpu_test, dense_pair_matches_reference) {
    if (!cm_qsa_ops_available()) {
        GTEST_SKIP() << "CM is not supported on this device";
    }
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    split_dims d;
    const size_t past_len = 20;
    const size_t q_len = 44;
    ASSERT_SELECTION_IS_DISCRIMINATING(d, past_len, q_len);

    auto ref = make_split_reference(rg, d, past_len, q_len);
    const size_t total = ref.total_len();  // S
    const size_t s_cap = total;
    const size_t summary_cap = s_cap / d.r;

    std::vector<float> raw_k_cache(s_cap * d.Di, 0.0f);
    std::vector<float> summary_cache(summary_cap * d.Di, 0.0f);
    std::vector<int32_t> position_cache(summary_cap, 0);
    for (size_t p = 0; p < past_len; p++) {
        for (size_t j = 0; j < d.Di; j++) {
            raw_k_cache[p * d.Di + j] = ref.k_idx_all[p * d.Di + j];
        }
    }
    for (size_t b = 0; b * d.r + d.r <= past_len; b++) {
        const auto s = ref.summary(b);
        for (size_t j = 0; j < d.Di; j++) {
            summary_cache[b * d.Di + j] = s[j];
        }
        // Trivial sequential position_ids convention, see the paged test's identical note.
        // One recorded position per block (its first token), same granularity as summary_cache.
        position_cache[b] = static_cast<int32_t>(b * d.r);
    }

    const std::vector<int32_t> past_lens = {static_cast<int32_t>(past_len)};
    std::vector<int32_t> position_ids(q_len);
    for (size_t t = 0; t < q_len; t++) {
        position_ids[t] = static_cast<int32_t>(past_len + t);
    }
    const auto cur_k_idx = std::vector<float>(ref.k_idx_all.begin() + past_len * d.Di, ref.k_idx_all.end());

    ov::intel_gpu::op::QSAIndexer::Config idx_cfg;
    idx_cfg.compress_ratio = d.r;
    idx_cfg.block_topk = d.block_topk;
    idx_cfg.h_sel = 1;
    idx_cfg.indexer_rotary_dim = d.rotary;
    idx_cfg.indexer_head_dim = d.Di;
    idx_cfg.eps = d.eps;
    idx_cfg.scale = 1.0f / std::sqrt(static_cast<float>(d.Di));

    ov::intel_gpu::op::SparseSDPA::Config attn_cfg;
    attn_cfg.num_heads = d.Hq;
    attn_cfg.num_kv_heads = d.Hkv;
    attn_cfg.h_sel = 1;
    attn_cfg.k_head_size = d.Dh;
    attn_cfg.v_head_size = d.Dv;
    attn_cfg.sel_block_size = d.r;
    attn_cfg.q_block_size = 1;
    attn_cfg.causal = true;

    topology topo;
    // B == 1 in this test; q_idx/k_idx/position_ids are reported rank-4/rank-2 per the
    // QSAIndexer RST but are byte-identical to the flat [tokens, ...] layout used above.
    topo.add(input_layout("q_idx", layout{{1, static_cast<int64_t>(q_len), static_cast<int64_t>(d.Hiq), static_cast<int64_t>(d.Di)}, data_types::f16,
                                          format::bfyx}));
    topo.add(input_layout("k_idx", layout{{1, static_cast<int64_t>(q_len), 1, static_cast<int64_t>(d.Di)}, data_types::f16, format::bfyx}));
    topo.add(input_layout("position_ids", layout{{1, static_cast<int64_t>(q_len)}, data_types::i32, format::bfyx}));
    topo.add(data("rotary", make_f16(engine, {static_cast<int64_t>(d.max_positions), static_cast<int64_t>(d.rotary)}, ref.rotary)));
    topo.add(input_layout("summary_cache", layout{{1, 1, static_cast<int64_t>(summary_cap), static_cast<int64_t>(d.Di)}, data_types::f16, format::bfyx}));
    topo.add(input_layout("raw_k_cache", layout{{1, 1, static_cast<int64_t>(s_cap), static_cast<int64_t>(d.Di)}, data_types::f16, format::bfyx}));
    topo.add(data("q_norm", make_f16(engine, {static_cast<int64_t>(d.Di)}, ref.q_norm)));
    topo.add(data("k_norm", make_f16(engine, {static_cast<int64_t>(d.Di)}, ref.k_norm)));
    topo.add(data("past_lens", make_i32(engine, {1}, past_lens)));
    topo.add(input_layout("position_cache", layout{{1, static_cast<int64_t>(summary_cap)}, data_types::i32, format::bfyx}));

    const std::vector<input_info> idx_inputs = {input_info("q_idx"),         input_info("k_idx"),     input_info("position_ids"),
                                                input_info("rotary"),       input_info("summary_cache"), input_info("raw_k_cache"),
                                                input_info("q_norm"),       input_info("k_norm"),     input_info("past_lens"),
                                                input_info("position_cache")};
    static_assert(DenseIdxIn::COUNT == 10, "keep the test input order in sync with QSAIndexer::InputIdx");
    topo.add(qsa_indexer("indexer", idx_inputs, idx_cfg));

    topo.add(input_layout("query", layout{{1, static_cast<int64_t>(d.Hq), static_cast<int64_t>(q_len), static_cast<int64_t>(d.Dh)}, data_types::f16,
                                          format::bfyx}));
    topo.add(input_layout("key", layout{{1, static_cast<int64_t>(d.Hkv), static_cast<int64_t>(total), static_cast<int64_t>(d.Dh)}, data_types::f16,
                                        format::bfyx}));
    topo.add(input_layout("value", layout{{1, static_cast<int64_t>(d.Hkv), static_cast<int64_t>(total), static_cast<int64_t>(d.Dv)}, data_types::f16,
                                          format::bfyx}));

    const std::vector<input_info> attn_inputs = {input_info("query"), input_info("key"), input_info("value"), input_info("indexer", 0),
                                                 input_info("indexer", 1)};
    static_assert(SparseSDPAIn::COUNT_MIN == 5, "keep the test input order in sync with SparseSDPA::InputIdx");
    topo.add(sparse_sdpa("attn", attn_inputs, attn_cfg, 1.0f / std::sqrt(static_cast<float>(d.Dh))));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::optimize_data(true));
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topo, cfg);

    network.set_input_data("q_idx", make_f16(engine, {1, static_cast<int64_t>(q_len), static_cast<int64_t>(d.Hiq), static_cast<int64_t>(d.Di)}, ref.q_idx_cur));
    network.set_input_data("k_idx", make_f16(engine, {1, static_cast<int64_t>(q_len), 1, static_cast<int64_t>(d.Di)}, cur_k_idx));
    network.set_input_data("position_ids", make_i32(engine, {1, static_cast<int64_t>(q_len)}, position_ids));
    network.set_input_data("summary_cache", make_f16(engine, {1, 1, static_cast<int64_t>(summary_cap), static_cast<int64_t>(d.Di)}, summary_cache));
    network.set_input_data("raw_k_cache", make_f16(engine, {1, 1, static_cast<int64_t>(s_cap), static_cast<int64_t>(d.Di)}, raw_k_cache));
    network.set_input_data("position_cache", make_i32(engine, {1, static_cast<int64_t>(summary_cap)}, position_cache));

    // Dense query/key/value use [B,H,*,D] layout -- transpose from the reference's
    // token-major [tokens, H*D] generation.
    std::vector<float> query_bhld(d.Hq * q_len * d.Dh);
    for (size_t t = 0; t < q_len; t++) {
        for (size_t h = 0; h < d.Hq; h++) {
            for (size_t j = 0; j < d.Dh; j++) {
                query_bhld[(h * q_len + t) * d.Dh + j] = ref.query_cur[(t * d.Hq + h) * d.Dh + j];
            }
        }
    }
    std::vector<float> key_bhsd(d.Hkv * total * d.Dh);
    std::vector<float> value_bhsd(d.Hkv * total * d.Dv);
    for (size_t p = 0; p < total; p++) {
        for (size_t h = 0; h < d.Hkv; h++) {
            for (size_t j = 0; j < d.Dh; j++) {
                key_bhsd[(h * total + p) * d.Dh + j] = ref.key_all[p * d.Hkv * d.Dh + h * d.Dh + j];
            }
            for (size_t j = 0; j < d.Dv; j++) {
                value_bhsd[(h * total + p) * d.Dv + j] = ref.value_all[p * d.Hkv * d.Dv + h * d.Dv + j];
            }
        }
    }
    network.set_input_data("query", make_f16(engine, {1, static_cast<int64_t>(d.Hq), static_cast<int64_t>(q_len), static_cast<int64_t>(d.Dh)}, query_bhld));
    network.set_input_data("key", make_f16(engine, {1, static_cast<int64_t>(d.Hkv), static_cast<int64_t>(total), static_cast<int64_t>(d.Dh)}, key_bhsd));
    network.set_input_data("value", make_f16(engine, {1, static_cast<int64_t>(d.Hkv), static_cast<int64_t>(total), static_cast<int64_t>(d.Dv)}, value_bhsd));

    auto outputs = network.execute();
    cldnn::mem_lock<ov::float16> out(outputs.at("attn").get_memory(), get_test_stream());

    std::vector<float> margins;
    const auto expected_token_major = ref.attention(&margins);  // [q_len, Hq*Dv] token-major

    size_t compared = 0;
    for (size_t t = 0; t < q_len; t++) {
        if (!std::isinf(margins[t]) && margins[t] < 0.02f) {
            continue;
        }
        compared++;
        for (size_t h = 0; h < d.Hq; h++) {
            for (size_t j = 0; j < d.Dv; j++) {
                const size_t out_idx = (h * q_len + t) * d.Dv + j;  // [B,Hq,L,Dv]
                const size_t exp_idx = (t * d.Hq + h) * d.Dv + j;
                EXPECT_NEAR(static_cast<float>(out[out_idx]), expected_token_major[exp_idx], 2e-2f) << "t=" << t << " h=" << h << " j=" << j;
            }
        }
    }
    ASSERT_GE(compared, q_len / 2) << "too many tokens were skipped as near-tied; test is not discriminating";
}

// --- host-level op validation (no GPU required) ----------------------------

std::shared_ptr<ov::op::v0::Parameter> make_param(const ov::PartialShape& shape, ov::element::Type_t et = ov::element::f16) {
    return std::make_shared<ov::op::v0::Parameter>(et, shape);
}

ov::OutputVector make_paged_indexer_args(const split_dims& d, size_t num_pages, size_t summaries_per_page) {
    return {make_param({-1, static_cast<int64_t>(d.Hiq * d.Di)}),
            make_param({-1, static_cast<int64_t>(d.Di)}),
            make_param({-1}, ov::element::i32),
            make_param({static_cast<int64_t>(d.max_positions), static_cast<int64_t>(d.rotary)}),
            make_param({static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(d.Di)}),
            make_param({static_cast<int64_t>(num_pages), 1, static_cast<int64_t>(d.page_size), static_cast<int64_t>(d.Di)}),
            make_param({static_cast<int64_t>(d.Di)}),
            make_param({static_cast<int64_t>(d.Di)}),
            make_param({-1}, ov::element::i32),
            make_param({-1}, ov::element::i32),
            make_param({-1}, ov::element::i32),
            make_param({-1}, ov::element::i32),
            make_param({static_cast<int64_t>(num_pages), static_cast<int64_t>(summaries_per_page)}, ov::element::i32)};
}

ov::intel_gpu::op::PagedQSAIndexer::Config make_paged_indexer_config(const split_dims& d) {
    ov::intel_gpu::op::PagedQSAIndexer::Config c;
    c.compress_ratio = d.r;
    c.block_topk = d.block_topk;
    c.h_sel = 1;
    c.indexer_rotary_dim = d.rotary;
    c.indexer_head_dim = d.Di;
    c.page_size = d.page_size;
    c.eps = d.eps;
    c.scale = 1.0f / std::sqrt(static_cast<float>(d.Di));
    return c;
}

TEST(qsa_ops_gpu_test, paged_qsa_indexer_config_validation_rejects_illegal_attributes) {
    std::string reason;
    split_dims d;
    auto base = make_paged_indexer_config(d);
    ASSERT_TRUE(ov::intel_gpu::op::PagedQSAIndexer::validate_config(base, reason)) << reason;

    auto c = base;
    c.compress_ratio = 0;
    ASSERT_FALSE(ov::intel_gpu::op::PagedQSAIndexer::validate_config(c, reason));

    c = base;
    c.block_topk = 0;
    ASSERT_FALSE(ov::intel_gpu::op::PagedQSAIndexer::validate_config(c, reason));

    c = base;
    c.indexer_rotary_dim = c.indexer_head_dim + 2;
    ASSERT_FALSE(ov::intel_gpu::op::PagedQSAIndexer::validate_config(c, reason));

    c = base;
    c.page_size = d.page_size + 1;  // not divisible by compress_ratio
    ASSERT_FALSE(ov::intel_gpu::op::PagedQSAIndexer::validate_config(c, reason));
}

TEST(qsa_ops_gpu_test, paged_qsa_indexer_output_shapes_and_clone) {
    split_dims d;
    const size_t num_pages = 8;
    const size_t summaries_per_page = d.page_size / d.r;
    auto node = std::make_shared<ov::intel_gpu::op::PagedQSAIndexer>(make_paged_indexer_args(d, num_pages, summaries_per_page), make_paged_indexer_config(d));

    ASSERT_EQ(node->get_output_size(), 2u);
    const auto& sel_indices_ps = node->get_output_partial_shape(0);
    ASSERT_EQ(sel_indices_ps.size(), 3u);
    ASSERT_EQ(sel_indices_ps[1].get_length(), 1);  // h_sel == 1
    ASSERT_EQ(sel_indices_ps[2].get_length(), static_cast<int64_t>(d.block_topk));
    ASSERT_EQ(node->get_output_element_type(0), ov::element::i32);
    ASSERT_EQ(node->get_output_element_type(1), ov::element::i32);

    auto clone = node->clone_with_new_inputs(node->input_values());
    ASSERT_NE(std::dynamic_pointer_cast<ov::intel_gpu::op::PagedQSAIndexer>(clone), nullptr)
        << "clone_with_new_inputs() must not silently downgrade to the base ov::op::internal::PagedQSAIndexer type";
}

TEST(qsa_ops_gpu_test, paged_qsa_indexer_rejects_wrong_input_count) {
    split_dims d;
    auto args = make_paged_indexer_args(d, 8, d.page_size / d.r);
    args.pop_back();
    ASSERT_THROW(std::make_shared<ov::intel_gpu::op::PagedQSAIndexer>(args, make_paged_indexer_config(d)), ov::Exception);
}

ov::intel_gpu::op::SparsePA::Config make_sparse_pa_config(const split_dims& d) {
    ov::intel_gpu::op::SparsePA::Config c;
    c.num_heads = d.Hq;
    c.num_kv_heads = d.Hkv;
    c.h_sel = 1;
    c.k_head_size = d.Dh;
    c.v_head_size = d.Dv;
    c.block_size = d.page_size;
    c.sel_block_size = d.r;
    c.q_block_size = 1;
    c.scale = 1.0f / std::sqrt(static_cast<float>(d.Dh));
    c.causal = true;
    return c;
}

TEST(qsa_ops_gpu_test, sparse_pa_config_validation_rejects_illegal_attributes) {
    std::string reason;
    split_dims d;
    auto base = make_sparse_pa_config(d);
    ASSERT_TRUE(ov::intel_gpu::op::SparsePA::validate_config(base, reason)) << reason;

    auto c = base;
    c.num_heads = 3;  // not a multiple of num_kv_heads
    ASSERT_FALSE(ov::intel_gpu::op::SparsePA::validate_config(c, reason));

    c = base;
    c.h_sel = 3;  // not 1, num_kv_heads, or num_heads
    ASSERT_FALSE(ov::intel_gpu::op::SparsePA::validate_config(c, reason));

    c = base;
    c.block_size = d.page_size + 1;  // not divisible by sel_block_size
    ASSERT_FALSE(ov::intel_gpu::op::SparsePA::validate_config(c, reason));

    c = base;
    c.q_block_size = 2;  // not implemented in this pass
    ASSERT_FALSE(ov::intel_gpu::op::SparsePA::validate_config(c, reason));
}

TEST(qsa_ops_gpu_test, sparse_pa_output_shape_and_clone) {
    split_dims d;
    const size_t num_pages = 8;
    ov::OutputVector args = {make_param({-1, static_cast<int64_t>(d.Hq * d.Dh)}),
                             make_param({-1, static_cast<int64_t>(d.Hkv * d.Dh)}),
                             make_param({-1, static_cast<int64_t>(d.Hkv * d.Dv)}),
                             make_param({static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.page_size),
                                        static_cast<int64_t>(d.Dh)}),
                             make_param({static_cast<int64_t>(num_pages), static_cast<int64_t>(d.Hkv), static_cast<int64_t>(d.page_size),
                                        static_cast<int64_t>(d.Dv)}),
                             make_param({-1}, ov::element::i32),
                             make_param({-1}, ov::element::i32),
                             make_param({-1}, ov::element::i32),
                             make_param({-1}, ov::element::i32),
                             make_param({-1, 1, static_cast<int64_t>(d.block_topk)}, ov::element::i32),
                             make_param({-1, 1}, ov::element::i32),
                             make_param({-1}, ov::element::i32)};
    auto node = std::make_shared<ov::intel_gpu::op::SparsePA>(args, make_sparse_pa_config(d));

    const auto& out_ps = node->get_output_partial_shape(0);
    ASSERT_EQ(out_ps.size(), 2u);
    ASSERT_EQ(out_ps[1].get_length(), static_cast<int64_t>(d.Hq * d.Dv));

    auto clone = node->clone_with_new_inputs(node->input_values());
    ASSERT_NE(std::dynamic_pointer_cast<ov::intel_gpu::op::SparsePA>(clone), nullptr);
}

ov::intel_gpu::op::SparseSDPA::Config make_sparse_sdpa_config(const split_dims& d) {
    ov::intel_gpu::op::SparseSDPA::Config c;
    c.num_heads = d.Hq;
    c.num_kv_heads = d.Hkv;
    c.h_sel = 1;
    c.k_head_size = d.Dh;
    c.v_head_size = d.Dv;
    c.sel_block_size = d.r;
    c.q_block_size = 1;
    c.causal = true;
    return c;
}

TEST(qsa_ops_gpu_test, sparse_sdpa_config_validation_rejects_illegal_attributes) {
    std::string reason;
    split_dims d;
    auto base = make_sparse_sdpa_config(d);
    ASSERT_TRUE(ov::intel_gpu::op::SparseSDPA::validate_config(base, reason)) << reason;

    auto c = base;
    c.h_sel = 3;
    ASSERT_FALSE(ov::intel_gpu::op::SparseSDPA::validate_config(c, reason));

    c = base;
    c.q_block_size = 2;
    ASSERT_FALSE(ov::intel_gpu::op::SparseSDPA::validate_config(c, reason));
}

TEST(qsa_ops_gpu_test, sparse_sdpa_output_shape_and_optional_inputs) {
    split_dims d;
    ov::OutputVector args = {make_param({1, static_cast<int64_t>(d.Hq), -1, static_cast<int64_t>(d.Dh)}),
                             make_param({1, static_cast<int64_t>(d.Hkv), -1, static_cast<int64_t>(d.Dh)}),
                             make_param({1, static_cast<int64_t>(d.Hkv), -1, static_cast<int64_t>(d.Dv)}),
                             make_param({1, 1, -1, static_cast<int64_t>(d.block_topk)}, ov::element::i32),
                             make_param({1, 1, -1}, ov::element::i32)};
    auto node = std::make_shared<ov::intel_gpu::op::SparseSDPA>(args, make_sparse_sdpa_config(d));
    ASSERT_FALSE(node->has_attn_mask());
    ASSERT_FALSE(node->has_scale_input());

    const auto& out_ps = node->get_output_partial_shape(0);
    ASSERT_EQ(out_ps.size(), 4u);
    ASSERT_EQ(out_ps[1].get_length(), static_cast<int64_t>(d.Hq));
    ASSERT_EQ(out_ps[3].get_length(), static_cast<int64_t>(d.Dv));

    args.push_back(make_param({1, 1, -1, -1}));  // attn_mask
    auto node_with_mask = std::make_shared<ov::intel_gpu::op::SparseSDPA>(args, make_sparse_sdpa_config(d));
    ASSERT_TRUE(node_with_mask->has_attn_mask());
    ASSERT_FALSE(node_with_mask->has_scale_input());

    auto clone = node->clone_with_new_inputs(node->input_values());
    ASSERT_NE(std::dynamic_pointer_cast<ov::intel_gpu::op::SparseSDPA>(clone), nullptr);
}

}  // namespace
