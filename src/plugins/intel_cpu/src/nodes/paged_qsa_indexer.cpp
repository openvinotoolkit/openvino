// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_qsa_indexer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_parallel.hpp"
#include "graph_context.h"
#include "kernels/scaled_attn/common.hpp"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::node {

namespace {

using InputIdx = ov::op::internal::PagedQSAIndexer::InputIdx;

/// @brief Fixed output order of ov::op::internal::PagedQSAIndexer (not part of the op's public API).
struct OutputIdx {
    static constexpr size_t SEL_INDICES = 0;  ///< [tokens, h_sel, block_topk]
    static constexpr size_t SEL_COUNT = 1;    ///< [tokens, h_sel]
};

float rms_inv_scale(const float* x, size_t n, float eps) {
    float sumsq = 0.0F;
    for (size_t i = 0; i < n; i++) {
        sumsq += x[i] * x[i];
    }
    return 1.0F / std::sqrt(sumsq / static_cast<float>(n) + eps);
}

// RMSNorm(src) * weight, followed by partial rotate-half RoPE over the first `rotary_dim`
// channels (pairs (j, j + rotary_dim/2)). `cos_sin_row` holds one interleaved (cos_d, sin_d) pair per
// rotary channel d (2 * rotary_dim floats), so the two halves of a pair may use different
// coefficients. Writes `dim` channels to `dst` (may alias `src`).
void rmsnorm_partial_rope(const float* src,
                          float* dst,
                          const float* weight,
                          size_t dim,
                          size_t rotary_dim,
                          float eps,
                          const float* cos_sin_row) {
    const float inv_rms = rms_inv_scale(src, dim, eps);
    for (size_t d = 0; d < dim; d++) {
        dst[d] = src[d] * inv_rms * weight[d];
    }
    const size_t half = rotary_dim / 2;
    for (size_t j = 0; j < half; j++) {
        const float cos_lo = cos_sin_row[2 * j];
        const float sin_lo = cos_sin_row[2 * j + 1];
        const float cos_hi = cos_sin_row[2 * (j + half)];
        const float sin_hi = cos_sin_row[2 * (j + half) + 1];
        const float x0 = dst[j];
        const float x1 = dst[j + half];
        dst[j] = x0 * cos_lo - x1 * sin_lo;
        dst[j + half] = x1 * cos_hi + x0 * sin_hi;
    }
}

// All floating-point data ports of PagedQSAIndexer (index_query, raw_index_keys, rope_cos, rope_sin,
// q_norm_weight, k_norm_weight, index_key_cache, index_block_rope_cache, block_index_keys_cache) may
// independently be negotiated as f32, f16, or bf16 -- these convert only the elements actually
// touched, so compute stays fp32 without paying for a whole-tensor Reorder on every call over
// caches that grow every step. Duplicated from qsa_indexer.cpp rather than factored into a
// shared header, matching this repo's convention for these small kernel-local helpers.
template <typename... Idx>
void load_row_f32(const PlainTensor& t, size_t len, float* dst, Idx... idx) {
    const auto prec = t.get_precision();
    if (prec == ov::element::bf16) {
        ov::Extensions::Cpu::XARCH::cvt_copy(dst, t.ptr<ov::bfloat16>(idx...), 1, len, 0, 0);
    } else if (prec == ov::element::f16) {
        ov::Extensions::Cpu::XARCH::cvt_copy(dst, t.ptr<ov::float16>(idx...), 1, len, 0, 0);
    } else {
        ov::Extensions::Cpu::XARCH::cvt_copy(dst, t.ptr<float>(idx...), 1, len, 0, 0);
    }
}

template <typename... Idx>
void store_row_f32(const PlainTensor& t, size_t len, const float* src, Idx... idx) {
    const auto prec = t.get_precision();
    if (prec == ov::element::bf16) {
        ov::Extensions::Cpu::XARCH::cvt_copy(t.ptr<ov::bfloat16>(idx...), src, 1, len, 0, 0);
    } else if (prec == ov::element::f16) {
        ov::Extensions::Cpu::XARCH::cvt_copy(t.ptr<ov::float16>(idx...), src, 1, len, 0, 0);
    } else {
        ov::Extensions::Cpu::XARCH::cvt_copy(t.ptr<float>(idx...), src, 1, len, 0, 0);
    }
}

}  // namespace

PagedQSAIndexer::PagedQSAIndexer(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto paged_qsa_indexer = ov::as_type_ptr<const ov::op::internal::PagedQSAIndexer>(op);
    OPENVINO_ASSERT(paged_qsa_indexer, "Expected PagedQSAIndexer node");
    m_config = paged_qsa_indexer->get_config();
}

void PagedQSAIndexer::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    // Every floating-point data port is independently echoed at whatever precision (f32/f16/
    // bf16) was negotiated upstream -- the reference kernel reads/writes all of them through
    // load_row_f32()/store_row_f32() rather than hardcoding f32, since index_key_cache/
    // index_block_rope_cache/block_index_keys_cache grow every step and forcing a whole-tensor Reorder
    // on every call would redo that conversion for the entire history each time. The page-table
    // and index inputs stay i32 -- reordering them to f32 would numerically convert the index
    // values instead of reinterpreting them. The two outputs are always i32.
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        const bool is_i32_port = i == InputIdx::PAST_LENS || i == InputIdx::SUBSEQUENCE_BEGINS ||
                                 i == InputIdx::BLOCK_INDICES || i == InputIdx::BLOCK_INDICES_BEGINS;
        const ov::element::Type precision = is_i32_port ? ov::element::i32 : getOriginalInputPrecisionAtPort(i);
        inPortConfigs.emplace_back(LayoutType::ncsp, precision, getInputShapeAtPort(i), false, -1);
    }
    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator{LayoutType::ncsp, ov::element::i32, getOutputShapeAtPort(OutputIdx::SEL_INDICES), false, -1},
        PortConfigurator{LayoutType::ncsp, ov::element::i32, getOutputShapeAtPort(OutputIdx::SEL_COUNT), false, -1}};
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref);
}

void PagedQSAIndexer::execute([[maybe_unused]] const dnnl::stream& strm) {
    execute_typed();
}

void PagedQSAIndexer::execute_typed() {
    using IndexT = int32_t;
    const auto& c = m_config;
    const auto r = static_cast<int64_t>(c.key_block_size);
    const size_t rotary_dim = c.indexer_rotary_dim;
    const size_t D_idx = c.indexer_head_dim;
    const float eps = c.epsilon;
    const float scale = ov::op::internal::PagedQSAIndexer::effective_scale(c);
    const size_t block_topk = c.block_topk;
    const size_t H_sel = c.num_selection_heads;

    PlainTensor q_idx(getSrcMemoryAtPort(InputIdx::INDEX_QUERY));
    PlainTensor k_idx(getSrcMemoryAtPort(InputIdx::RAW_INDEX_KEYS));
    PlainTensor rope_cos(getSrcMemoryAtPort(InputIdx::ROPE_COS));
    PlainTensor rope_sin(getSrcMemoryAtPort(InputIdx::ROPE_SIN));
    PlainTensor q_norm_weight(getSrcMemoryAtPort(InputIdx::Q_NORM_WEIGHT));
    PlainTensor k_norm_weight(getSrcMemoryAtPort(InputIdx::K_NORM_WEIGHT));
    PlainTensor past_lens(getSrcMemoryAtPort(InputIdx::PAST_LENS));
    PlainTensor subsequence_begins(getSrcMemoryAtPort(InputIdx::SUBSEQUENCE_BEGINS));
    PlainTensor block_indices(getSrcMemoryAtPort(InputIdx::BLOCK_INDICES));
    PlainTensor block_indices_begins(getSrcMemoryAtPort(InputIdx::BLOCK_INDICES_BEGINS));
    PlainTensor raw_k_cache(getSrcMemoryAtPort(InputIdx::INDEX_KEY_CACHE));
    PlainTensor indexer_rope_cache(getSrcMemoryAtPort(InputIdx::INDEX_BLOCK_ROPE_CACHE));
    PlainTensor summary_cache(getSrcMemoryAtPort(InputIdx::BLOCK_INDEX_KEYS_CACHE));
    PlainTensor sel_indices(getDstMemoryAtPort(OutputIdx::SEL_INDICES));
    PlainTensor sel_count(getDstMemoryAtPort(OutputIdx::SEL_COUNT));

    // page_size is carried by the cache shape: [N_phys, H_ik, page_size, D_idx]
    const auto page_size = static_cast<int64_t>(raw_k_cache.size(2));
    const int64_t slots_per_page = page_size / r;

    const size_t tokens = q_idx.size(0);
    const size_t H_iq = q_idx.size(1) / D_idx;
    const size_t H_ik = k_idx.size(1) / D_idx;
    const size_t G = H_iq / H_ik;

    const auto* subsequence_begins_p = subsequence_begins.ptr<int32_t>();
    const auto* block_indices_begins_p = block_indices_begins.ptr<int32_t>();
    const auto* past_lens_p = past_lens.ptr<int32_t>();

    const auto seq_tensor_size = subsequence_begins.size(0);
    OPENVINO_ASSERT(seq_tensor_size >= 1, "[CPU] paged_qsa_indexer: subsequence_begins must be non-empty");
    const size_t num_sequences = seq_tensor_size - 1;

    std::vector<float> q_norm_weight_f32(D_idx);
    load_row_f32(q_norm_weight, D_idx, q_norm_weight_f32.data());
    std::vector<float> k_norm_weight_f32(D_idx);
    load_row_f32(k_norm_weight, D_idx, k_norm_weight_f32.data());

    std::vector<size_t> kv_of_head(H_iq);
    std::vector<size_t> sel_of_head(H_iq);
    for (size_t h = 0; h < H_iq; h++) {
        kv_of_head[h] = h / G;
        sel_of_head[h] = (kv_of_head[h] * H_sel) / H_ik;
    }

    std::fill_n(sel_indices.ptr<IndexT>(), tokens * H_sel * block_topk, static_cast<IndexT>(-1));

    const auto& cpu_parallel = context->getCpuParallel();
    cpu_parallel->parallel_for(num_sequences, [&](size_t seq) {
        const int32_t token_begin = subsequence_begins_p[seq];
        const int32_t token_end = subsequence_begins_p[seq + 1];
        const int32_t block_begin = block_indices_begins_p[seq];
        const int32_t block_end = block_indices_begins_p[seq + 1];
        const int32_t seq_blocks = std::max(block_end - block_begin, 0);
        const int64_t seq_past_len = past_lens_p[seq];
        OPENVINO_ASSERT(seq_blocks > 0, "[CPU] paged_qsa_indexer: each sequence must have at least one cache block");

        // Maps a logical token position (relative to this sequence's own 0..S history) to the
        // physical (block, slot) address into INDEX_KEY_CACHE through the shared page table.
        auto token_phys = [&](int64_t logical_pos) {
            const int64_t phys_idx = block_begin + logical_pos / page_size;
            const auto phys_block = static_cast<size_t>(*block_indices.ptr<int32_t>(static_cast<size_t>(phys_idx)));
            const auto slot = static_cast<size_t>(logical_pos % page_size);
            return std::make_pair(phys_block, slot);
        };
        // Maps a (sequence-local, 0-based) completed-block index to the physical (block, slot)
        // address into BLOCK_INDEX_KEYS_CACHE / INDEX_BLOCK_ROPE_CACHE.
        auto block_phys = [&](int64_t block_idx) {
            const int64_t raw_pos = block_idx * r;
            const int64_t phys_idx = block_begin + raw_pos / page_size;
            const auto phys_block = static_cast<size_t>(*block_indices.ptr<int32_t>(static_cast<size_t>(phys_idx)));
            const auto slot = static_cast<size_t>(block_idx % slots_per_page);
            return std::make_pair(phys_block, slot);
        };
        const int64_t block_capacity = static_cast<int64_t>(seq_blocks) * slots_per_page;

        std::vector<float> q_normed(H_iq * D_idx);
        std::vector<float> row_buf(D_idx);
        std::vector<float> rope_cos_buf(rotary_dim);
        std::vector<float> rope_sin_buf(rotary_dim);
        std::vector<float> cos_sin_buf(2 * rotary_dim);
        std::vector<float> cos_sin_block_buf(2 * rotary_dim);
        std::vector<float> pooled(D_idx);
        std::vector<float> summary_buf(D_idx);
        std::vector<std::pair<float, int32_t>> scores;
        std::vector<int32_t> chosen;

        for (int32_t token = token_begin; token < token_end; token++) {
            const size_t token_u = static_cast<size_t>(token);
            const int64_t logical_pos = seq_past_len + (token - token_begin);

            // ---- Stage A: query RMSNorm + partial RoPE (per index-query head) ----
            load_row_f32(rope_cos, rotary_dim, rope_cos_buf.data(), token_u);
            load_row_f32(rope_sin, rotary_dim, rope_sin_buf.data(), token_u);
            for (size_t j = 0; j < rotary_dim; j++) {
                cos_sin_buf[2 * j] = rope_cos_buf[j];
                cos_sin_buf[2 * j + 1] = rope_sin_buf[j];
            }
            if (logical_pos % r == 0) {
                // This token starts a new block: record its interleaved cos/sin row directly, so
                // the block's summary (finalized once the block completes, possibly in a later
                // call) can be RoPE'ed with the exact coefficients of its first token, regardless
                // of KV-cache slot vs logical position drift (continuous batching, chunked
                // prefill, mRoPE, ...). The bounds check skips a trailing block that starts within
                // the allocated page range but has no room to ever complete.
                const int64_t block_idx = logical_pos / r;
                if (block_idx < block_capacity) {
                    const auto [phys_block, slot] = block_phys(block_idx);
                    store_row_f32(indexer_rope_cache, 2 * rotary_dim, cos_sin_buf.data(), phys_block, slot);
                }
            }
            for (size_t h = 0; h < H_iq; h++) {
                load_row_f32(q_idx, D_idx, row_buf.data(), token_u, h * D_idx);
                rmsnorm_partial_rope(row_buf.data(),
                                     &q_normed[h * D_idx],
                                     q_norm_weight_f32.data(),
                                     D_idx,
                                     rotary_dim,
                                     eps,
                                     cos_sin_buf.data());
            }

            // ---- Stage B: raw-K append + block summary pooling (per index-key head) ----
            const auto [cur_phys_block, cur_slot] = token_phys(logical_pos);
            for (size_t h = 0; h < H_ik; h++) {
                load_row_f32(k_idx, D_idx, row_buf.data(), token_u, h * D_idx);
                store_row_f32(raw_k_cache, D_idx, row_buf.data(), cur_phys_block, h, cur_slot);
            }
            const int64_t n_complete = (logical_pos + 1) / r;
            if ((logical_pos + 1) % r == 0) {
                const int64_t block_idx = n_complete - 1;
                const int64_t block_start = block_idx * r;
                const auto [block_phys_block, block_slot] = block_phys(block_idx);
                load_row_f32(indexer_rope_cache, 2 * rotary_dim, cos_sin_block_buf.data(), block_phys_block, block_slot);
                for (size_t h = 0; h < H_ik; h++) {
                    std::fill(pooled.begin(), pooled.end(), 0.0F);
                    for (int64_t p = block_start; p < block_start + r; p++) {
                        const auto [p_phys_block, p_slot] = token_phys(p);
                        load_row_f32(raw_k_cache, D_idx, row_buf.data(), p_phys_block, h, p_slot);
                        for (size_t d = 0; d < D_idx; d++) {
                            pooled[d] += row_buf[d];
                        }
                    }
                    const float inv_r = 1.0F / static_cast<float>(r);
                    for (size_t d = 0; d < D_idx; d++) {
                        pooled[d] *= inv_r;
                    }
                    rmsnorm_partial_rope(pooled.data(),
                                         summary_buf.data(),
                                         k_norm_weight_f32.data(),
                                         D_idx,
                                         rotary_dim,
                                         eps,
                                         cos_sin_block_buf.data());
                    store_row_f32(summary_cache, D_idx, summary_buf.data(), block_phys_block, h, block_slot);
                }
            }

            // ---- Stage C: per-selection-head complete-block scoring + top-k ----
            for (size_t sh = 0; sh < H_sel; sh++) {
                size_t k = 0;
                if (n_complete > 0) {
                    scores.resize(static_cast<size_t>(n_complete));
                    for (int64_t bidx = 0; bidx < n_complete; bidx++) {
                        const auto [s_phys_block, s_slot] = block_phys(bidx);
                        float score = 0.0F;
                        for (size_t h = 0; h < H_iq; h++) {
                            if (sel_of_head[h] != sh) {
                                continue;
                            }
                            load_row_f32(summary_cache, D_idx, summary_buf.data(), s_phys_block, kv_of_head[h], s_slot);
                            float dot = 0.0F;
                            for (size_t d = 0; d < D_idx; d++) {
                                dot += q_normed[h * D_idx + d] * summary_buf[d];
                            }
                            score += std::max(0.0F, dot);
                        }
                        scores[static_cast<size_t>(bidx)] = {score * scale, static_cast<int32_t>(bidx)};
                    }
                    k = std::min(block_topk, static_cast<size_t>(n_complete));
                    std::partial_sort(scores.begin(),
                                      scores.begin() + static_cast<ptrdiff_t>(k),
                                      scores.end(),
                                      [](const auto& a, const auto& b2) {
                                          if (a.first != b2.first) {
                                              return a.first > b2.first;
                                          }
                                          return a.second < b2.second;
                                      });
                    chosen.resize(k);
                    for (size_t i = 0; i < k; i++) {
                        chosen[i] = scores[i].second;
                    }
                    std::sort(chosen.begin(), chosen.end());
                }

                auto* out_row = sel_indices.ptr<IndexT>(token_u, sh, 0);
                for (size_t i = 0; i < k; i++) {
                    out_row[i] = static_cast<IndexT>(chosen[i]);
                }
                *sel_count.ptr<IndexT>(token_u, sh) = static_cast<IndexT>(k);
            }
        }
    });
}

bool PagedQSAIndexer::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                           std::string& errorMessage) noexcept {
    try {
        const auto paged_qsa_indexer = ov::as_type_ptr<const ov::op::internal::PagedQSAIndexer>(op);
        if (!paged_qsa_indexer) {
            errorMessage = "Only PagedQSAIndexer operation is supported";
            return false;
        }
        static constexpr size_t float_ports[] = {InputIdx::INDEX_QUERY,
                                                 InputIdx::RAW_INDEX_KEYS,
                                                 InputIdx::ROPE_COS,
                                                 InputIdx::ROPE_SIN,
                                                 InputIdx::Q_NORM_WEIGHT,
                                                 InputIdx::K_NORM_WEIGHT,
                                                 InputIdx::INDEX_KEY_CACHE,
                                                 InputIdx::INDEX_BLOCK_ROPE_CACHE,
                                                 InputIdx::BLOCK_INDEX_KEYS_CACHE};
        for (auto port : float_ports) {
            const auto prec = op->get_input_element_type(port);
            if (prec != ov::element::f32 && prec != ov::element::f16 && prec != ov::element::bf16) {
                errorMessage = "PagedQSAIndexer CPU implementation only supports f32, f16, or bf16 data inputs.";
                return false;
            }
        }
        static constexpr size_t i32_ports[] = {InputIdx::PAST_LENS,
                                               InputIdx::SUBSEQUENCE_BEGINS,
                                               InputIdx::BLOCK_INDICES,
                                               InputIdx::BLOCK_INDICES_BEGINS};
        for (auto port : i32_ports) {
            if (op->get_input_element_type(port) != ov::element::i32) {
                errorMessage = "PagedQSAIndexer CPU implementation requires i32 index inputs.";
                return false;
            }
        }
        std::string reason;
        if (!ov::op::internal::PagedQSAIndexer::validate_config(paged_qsa_indexer->get_config(), reason)) {
            errorMessage = "PagedQSAIndexer configuration is invalid: " + reason;
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
