// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_indexer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
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
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::node {

namespace {

using InputIdx = ov::intel_cpu::QSAIndexer::InputIdx;
using OutputIdx = ov::intel_cpu::QSAIndexer::OutputIdx;

float rms_inv_scale(const float* x, size_t n, float eps) {
    float sumsq = 0.0F;
    for (size_t i = 0; i < n; i++) {
        sumsq += x[i] * x[i];
    }
    return 1.0F / std::sqrt(sumsq / static_cast<float>(n) + eps);
}

// RMSNorm(src) * weight, followed by the split-half (rotate-half) RoPE of HF Transformers over
// the first `rotary_dim` channels: x * cos + rotate_half(x) * sin, with
// rotate_half(x) = concat(-x[half:], x[:half]). `cos_row`/`sin_row` each hold `rotary_dim`
// values (not halved -- the caller's cos/sin are already duplicated across the two halves, per
// new_spec/qsa-prepare-17.rst). Writes `dim` channels to `dst` (may alias `src`).
void rmsnorm_partial_rope(const float* src,
                          float* dst,
                          const float* weight,
                          size_t dim,
                          size_t rotary_dim,
                          float eps,
                          const float* cos_row,
                          const float* sin_row) {
    const float inv_rms = rms_inv_scale(src, dim, eps);
    for (size_t d = 0; d < dim; d++) {
        dst[d] = src[d] * inv_rms * weight[d];
    }
    const size_t half = rotary_dim / 2;
    for (size_t j = 0; j < half; j++) {
        const float x0 = dst[j];
        const float x1 = dst[j + half];
        dst[j] = x0 * cos_row[j] - x1 * sin_row[j];
        dst[j + half] = x1 * cos_row[j + half] + x0 * sin_row[j + half];
    }
}

// All floating-point data ports of QSAIndexer may independently be negotiated as f32, f16, or
// bf16 -- these convert only the elements actually touched, so compute stays fp32 without paying
// for a whole-tensor Reorder on every call. Reuses the same SIMD converting-copy primitive as
// recurrent_linear_attn_paged_impl (kernels/scaled_attn/common.hpp).
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

// Output shapes of QSAIndexer: everything follows the input shapes, but the summary_cache block count of the longest
// row, which is read from the key_length values.
class QSAIndexerShapeInfer : public ShapeInferEmptyPads {
public:
    explicit QSAIndexerShapeInfer(const ov::intel_cpu::QSAIndexer::Config& config) : m_config(config) {}

    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const auto& q_dims = input_shapes[InputIdx::Q_IDX].get();
        const auto& summary_dims = input_shapes[InputIdx::SUMMARY_CACHE].get();
        const size_t batch = q_dims[0];
        const auto key_length = data_dependency.at(InputIdx::KEY_LENGTH);
        const auto* key_length_p = key_length->getDataAs<const int32_t>();
        int64_t longest = 0;
        for (size_t b = 0; b < batch; b++) {
            longest = std::max<int64_t>(longest, key_length_p[b]);
        }
        const auto num_blocks = static_cast<size_t>(longest / static_cast<int64_t>(m_config.compress_ratio));
        return {{{batch, m_config.h_sel, q_dims[1], m_config.block_topk},
                 {batch, m_config.h_sel, q_dims[1]},
                 {batch, summary_dims[1], num_blocks, summary_dims[3]},
                 input_shapes[InputIdx::PAST_INCOMPLETE_RAW_K].get(),
                 input_shapes[InputIdx::PAST_INCOMPLETE_ROPE].get()},
                ShapeInferStatus::success};
    }

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return 1U << InputIdx::KEY_LENGTH;
    }

private:
    ov::intel_cpu::QSAIndexer::Config m_config;
};

class QSAIndexerShapeInferFactory : public ShapeInferFactory {
public:
    explicit QSAIndexerShapeInferFactory(const std::shared_ptr<ov::Node>& op) {
        if (const auto qsa_indexer = ov::as_type_ptr<const ov::intel_cpu::QSAIndexer>(op)) {
            m_config = qsa_indexer->get_config();
        }
    }

    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<QSAIndexerShapeInfer>(m_config);
    }

private:
    ov::intel_cpu::QSAIndexer::Config m_config;
};

}  // namespace

QSAIndexer::QSAIndexer(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, QSAIndexerShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto qsa_indexer = ov::as_type_ptr<const ov::intel_cpu::QSAIndexer>(op);
    OPENVINO_ASSERT(qsa_indexer, "Expected QSAIndexer node");
    m_config = qsa_indexer->get_config();
}

void QSAIndexer::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    // Every floating-point data port is independently echoed at whatever precision (f32/f16/
    // bf16) was negotiated upstream -- the reference kernel reads/writes all of them through
    // load_row_f32()/store_row_f32() rather than hardcoding f32. KEY_LENGTH stays i32 --
    // reordering it to f32 would numerically convert the index values instead of reinterpreting
    // them. sel_indices/sel_count are always i32; each state output has the precision of its
    // state input and never shares memory with it (inPlace == -1): the update is out-of-place.
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        const bool is_i32_port = i == InputIdx::KEY_LENGTH;
        const ov::element::Type precision = is_i32_port ? ov::element::i32 : getOriginalInputPrecisionAtPort(i);
        inPortConfigs.emplace_back(LayoutType::ncsp, precision, getInputShapeAtPort(i), false, -1);
    }
    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator{LayoutType::ncsp, ov::element::i32, getOutputShapeAtPort(OutputIdx::SEL_INDICES), false, -1},
        PortConfigurator{LayoutType::ncsp, ov::element::i32, getOutputShapeAtPort(OutputIdx::SEL_COUNT), false, -1}};
    static constexpr std::pair<size_t, size_t> state_ports[] = {
        {InputIdx::SUMMARY_CACHE, OutputIdx::SUMMARY_CACHE},
        {InputIdx::PAST_INCOMPLETE_RAW_K, OutputIdx::PAST_INCOMPLETE_RAW_K},
        {InputIdx::PAST_INCOMPLETE_ROPE, OutputIdx::PAST_INCOMPLETE_ROPE}};
    for (const auto& [in_port, out_port] : state_ports) {
        outPortConfigs.emplace_back(LayoutType::ncsp,
                                    getOriginalInputPrecisionAtPort(in_port),
                                    getOutputShapeAtPort(out_port),
                                    false,
                                    -1);
    }
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref);
}

void QSAIndexer::execute([[maybe_unused]] const dnnl::stream& strm) {
    execute_typed<int32_t>();
}

template <typename IndexT>
void QSAIndexer::execute_typed() {
    const auto& c = m_config;
    const auto r = static_cast<int64_t>(c.compress_ratio);
    const size_t rotary_dim = c.indexer_rotary_dim;
    const float eps = c.eps;
    const float scale = ov::intel_cpu::QSAIndexer::effective_scale(c);
    const size_t block_topk = c.block_topk;
    const size_t H_sel = c.h_sel;

    PlainTensor q_idx(getSrcMemoryAtPort(InputIdx::Q_IDX));
    PlainTensor k_idx(getSrcMemoryAtPort(InputIdx::K_IDX));
    PlainTensor rope_cos(getSrcMemoryAtPort(InputIdx::ROPE_COS));
    PlainTensor rope_sin(getSrcMemoryAtPort(InputIdx::ROPE_SIN));
    PlainTensor key_length(getSrcMemoryAtPort(InputIdx::KEY_LENGTH));
    PlainTensor q_norm_weight(getSrcMemoryAtPort(InputIdx::Q_NORM_WEIGHT));
    PlainTensor k_norm_weight(getSrcMemoryAtPort(InputIdx::K_NORM_WEIGHT));
    // State inputs are read-only; the updated state lives in the matching outputs.
    PlainTensor past_incomplete_raw_k_in(getSrcMemoryAtPort(InputIdx::PAST_INCOMPLETE_RAW_K));
    PlainTensor past_incomplete_rope_in(getSrcMemoryAtPort(InputIdx::PAST_INCOMPLETE_ROPE));
    PlainTensor summary_cache(getDstMemoryAtPort(OutputIdx::SUMMARY_CACHE));
    PlainTensor past_incomplete_raw_k(getDstMemoryAtPort(OutputIdx::PAST_INCOMPLETE_RAW_K));
    PlainTensor past_incomplete_rope(getDstMemoryAtPort(OutputIdx::PAST_INCOMPLETE_ROPE));
    PlainTensor sel_indices(getDstMemoryAtPort(OutputIdx::SEL_INDICES));
    PlainTensor sel_count(getDstMemoryAtPort(OutputIdx::SEL_COUNT));

    const size_t B = q_idx.size(0);
    const int64_t L = static_cast<int64_t>(q_idx.size(1));
    const size_t H_iq = q_idx.size(2);
    const size_t D_idx = q_idx.size(3);
    const size_t H_ik = k_idx.size(2);
    const size_t G = H_iq / H_ik;

    const auto* key_length_p = key_length.ptr<int32_t>();

    // summary_cache is the only state that grows: its output starts as a copy of the blocks of the earlier calls
    // (the input), followed by zeros, and the blocks completed by this call are written over them below.
    const auto summary_cache_in_mem = getSrcMemoryAtPort(InputIdx::SUMMARY_CACHE);
    const auto summary_cache_out_mem = getDstMemoryAtPort(OutputIdx::SUMMARY_CACHE);
    const size_t summary_blocks_in = summary_cache_in_mem->getStaticDims()[2];
    const size_t summary_blocks_out = summary_cache_out_mem->getStaticDims()[2];
    const size_t summary_block_bytes = D_idx * summary_cache_in_mem->getDesc().getPrecision().size();
    const size_t summary_head_in_bytes = summary_blocks_in * summary_block_bytes;
    const size_t summary_head_out_bytes = summary_blocks_out * summary_block_bytes;
    const size_t summary_copy_bytes = std::min(summary_blocks_in, summary_blocks_out) * summary_block_bytes;
    const auto* summary_cache_src = summary_cache_in_mem->getDataAs<const uint8_t>();
    auto* summary_cache_dst = summary_cache_out_mem->getDataAs<uint8_t>();

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

    std::fill_n(sel_indices.ptr<IndexT>(), B * H_sel * static_cast<size_t>(L) * block_topk, static_cast<IndexT>(-1));
    std::fill_n(sel_count.ptr<IndexT>(), B * H_sel * static_cast<size_t>(L), static_cast<IndexT>(0));

    const auto& cpu_parallel = context->getCpuParallel();
    cpu_parallel->parallel_for(B, [&](size_t b) {
        for (size_t h = 0; h < H_ik; h++) {
            auto* dst = summary_cache_dst + (b * H_ik + h) * summary_head_out_bytes;
            const auto* src = summary_cache_src + (b * H_ik + h) * summary_head_in_bytes;
            if (summary_copy_bytes != 0) {
                std::memcpy(dst, src, summary_copy_bytes);
            }
            if (summary_head_out_bytes != summary_copy_bytes) {
                std::memset(dst + summary_copy_bytes, 0, summary_head_out_bytes - summary_copy_bytes);
            }
        }

        const int64_t key_len = key_length_p[b];
        // Every per-row quantity is derived from this call's own key_length/L alone -- no
        // cross-call running counter is kept (see QSAIndexer's class comment).
        const int64_t past_n = std::max<int64_t>(0, key_len - L);
        const int64_t current_n = key_len - past_n;  // == min(L, key_len)
        const int64_t num_padding_n = L - current_n;
        const int64_t c_n = past_n % r;  // valid rows already sitting in the tail buffer
        const int64_t num_cached_blocks_n = past_n / r;

        std::vector<float> q_normed(H_iq * D_idx);
        std::vector<float> row_buf(D_idx);
        std::vector<float> rope_cos_buf(rotary_dim);
        std::vector<float> rope_sin_buf(rotary_dim);
        // rope_cos (first rotary_dim) then rope_sin (next rotary_dim) of the block's reference
        // token, matching past_incomplete_rope's [rope_cos, rope_sin] layout 1:1.
        std::vector<float> block_ref_cos_sin(2 * rotary_dim, 0.0F);
        std::vector<float> pooled(D_idx);
        std::vector<float> summary_buf(D_idx);
        std::vector<std::pair<float, int32_t>> scores;
        std::vector<int32_t> chosen;

        // Raw rows of the block currently being accumulated (0..r-1 valid slots), seeded from
        // the caller-owned fixed-size tail buffer and/or this call's new K_IDX rows.
        std::vector<float> pending_rows(static_cast<size_t>(r) * H_ik * D_idx, 0.0F);
        auto pending_row_ptr = [&](int64_t pos_in_block, size_t h) {
            return pending_rows.data() + (static_cast<size_t>(pos_in_block) * H_ik + h) * D_idx;
        };

        for (int64_t p = 0; p < c_n; p++) {
            for (size_t h = 0; h < H_ik; h++) {
                load_row_f32(past_incomplete_raw_k_in, D_idx, pending_row_ptr(p, h), b, h, p);
            }
        }
        if (c_n > 0) {
            load_row_f32(past_incomplete_rope_in, 2 * rotary_dim, block_ref_cos_sin.data(), b);
        }

        int64_t running_block_idx = num_cached_blocks_n;

        for (int64_t l = num_padding_n; l < L; l++) {
            const int64_t v = l - num_padding_n;
            const int64_t pending_pos = c_n + v;
            const int64_t pos_in_block = pending_pos % r;

            // ---- Stage A: query RMSNorm + partial RoPE (per index-query head) ----
            load_row_f32(rope_cos, rotary_dim, rope_cos_buf.data(), b, l);
            load_row_f32(rope_sin, rotary_dim, rope_sin_buf.data(), b, l);
            for (size_t h = 0; h < H_iq; h++) {
                load_row_f32(q_idx, D_idx, row_buf.data(), b, l, h);
                rmsnorm_partial_rope(row_buf.data(),
                                     &q_normed[h * D_idx],
                                     q_norm_weight_f32.data(),
                                     D_idx,
                                     rotary_dim,
                                     eps,
                                     rope_cos_buf.data(),
                                     rope_sin_buf.data());
            }

            // ---- Stage B: raw-K append into the in-progress block + pooling on completion ----
            if (pos_in_block == 0) {
                // c_n < r guarantees pending_pos == 0 only ever occurs when c_n == 0, i.e. this is
                // genuinely a fresh block (not one seeded from the tail) -- safe to overwrite.
                std::copy(rope_cos_buf.begin(), rope_cos_buf.end(), block_ref_cos_sin.begin());
                std::copy(rope_sin_buf.begin(),
                          rope_sin_buf.end(),
                          block_ref_cos_sin.begin() + static_cast<ptrdiff_t>(rotary_dim));
            }
            for (size_t h = 0; h < H_ik; h++) {
                load_row_f32(k_idx, D_idx, pending_row_ptr(pos_in_block, h), b, l, h);
            }

            if (pos_in_block + 1 == r) {
                for (size_t h = 0; h < H_ik; h++) {
                    std::fill(pooled.begin(), pooled.end(), 0.0F);
                    for (int64_t p = 0; p < r; p++) {
                        const float* src = pending_row_ptr(p, h);
                        for (size_t d = 0; d < D_idx; d++) {
                            pooled[d] += src[d];
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
                                         block_ref_cos_sin.data(),
                                         block_ref_cos_sin.data() + rotary_dim);
                    OPENVINO_ASSERT(static_cast<size_t>(running_block_idx) < summary_blocks_out,
                                    "QSAIndexer: block ",
                                    running_block_idx,
                                    " does not fit the summary_cache output of ",
                                    summary_blocks_out,
                                    " blocks");
                    store_row_f32(summary_cache, D_idx, summary_buf.data(), b, h, running_block_idx);
                }
                running_block_idx += 1;
            }

            // ---- Stage C: per-selection-head complete-block scoring + top-k ----
            const int64_t n_complete = running_block_idx;
            for (size_t sh = 0; sh < H_sel; sh++) {
                size_t k = 0;
                if (n_complete > 0) {
                    scores.resize(static_cast<size_t>(n_complete));
                    for (int64_t bidx = 0; bidx < n_complete; bidx++) {
                        float score = 0.0F;
                        for (size_t h = 0; h < H_iq; h++) {
                            if (sel_of_head[h] != sh) {
                                continue;
                            }
                            load_row_f32(summary_cache, D_idx, summary_buf.data(), b, kv_of_head[h], bidx);
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

                auto* out_row = sel_indices.ptr<IndexT>(b, sh, l, 0);
                for (size_t i = 0; i < k; i++) {
                    out_row[i] = static_cast<IndexT>(chosen[i]);
                }
                *sel_count.ptr<IndexT>(b, sh, l) = static_cast<IndexT>(k);
            }
        }

        // ---- Finalize: fully overwrite the fixed-size incomplete-block tail buffers ----
        const int64_t rest = (c_n + current_n) - (running_block_idx - num_cached_blocks_n) * r;
        std::vector<float> zero_row(D_idx, 0.0F);
        for (int64_t p = 0; p < r - 1; p++) {
            for (size_t h = 0; h < H_ik; h++) {
                const float* src = p < rest ? pending_row_ptr(p, h) : zero_row.data();
                store_row_f32(past_incomplete_raw_k, D_idx, src, b, h, p);
            }
        }
        if (rest == 0) {
            std::fill(block_ref_cos_sin.begin(), block_ref_cos_sin.end(), 0.0F);
        }
        store_row_f32(past_incomplete_rope, 2 * rotary_dim, block_ref_cos_sin.data(), b);
    });
}

bool QSAIndexer::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto qsa_indexer = ov::as_type_ptr<const ov::intel_cpu::QSAIndexer>(op);
        if (!qsa_indexer) {
            errorMessage = "Only QSAIndexer operation is supported";
            return false;
        }
        static constexpr size_t float_ports[] = {InputIdx::Q_IDX,
                                                 InputIdx::K_IDX,
                                                 InputIdx::ROPE_COS,
                                                 InputIdx::ROPE_SIN,
                                                 InputIdx::Q_NORM_WEIGHT,
                                                 InputIdx::K_NORM_WEIGHT,
                                                 InputIdx::SUMMARY_CACHE,
                                                 InputIdx::PAST_INCOMPLETE_RAW_K,
                                                 InputIdx::PAST_INCOMPLETE_ROPE};
        for (auto port : float_ports) {
            const auto prec = op->get_input_element_type(port);
            if (prec != ov::element::f32 && prec != ov::element::f16 && prec != ov::element::bf16) {
                errorMessage = "QSAIndexer CPU implementation only supports f32, f16, or bf16 data inputs.";
                return false;
            }
        }
        static constexpr size_t i32_ports[] = {InputIdx::KEY_LENGTH};
        for (auto port : i32_ports) {
            if (op->get_input_element_type(port) != ov::element::i32) {
                errorMessage = "QSAIndexer CPU implementation requires i32 index inputs.";
                return false;
            }
        }
        std::string reason;
        if (!ov::intel_cpu::QSAIndexer::validate_config(qsa_indexer->get_config(), reason)) {
            errorMessage = "QSAIndexer configuration is invalid: " + reason;
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
