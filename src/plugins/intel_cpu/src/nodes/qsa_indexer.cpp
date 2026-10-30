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
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_parallel.hpp"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "node.h"
#include "nodes/common/cpu_convert.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"
#include "utils/general_utils.h"
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

// All floating-point ports share the precision T (the node declares them that way), so only the rows
// actually touched are converted to / from the f32 compute buffers, never a whole tensor.
// Read-only f32 view of a row: the tensor row itself when T is f32 (nothing to convert, so no copy), otherwise the
// row converted into `scratch`. Rows that have to be copied anyway call cpu_convert directly.
template <typename T, typename... Idx>
const float* row_f32(const PlainTensor& t, size_t len, float* scratch, Idx... idx) {
    if constexpr (std::is_same_v<T, float>) {
        return t.ptr<float>(idx...);
    } else {
        cpu_convert(t.ptr<T>(idx...), scratch, ov::element::from<T>(), ov::element::f32, len);
        return scratch;
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
        // key_length is [batch] or [1] (the same length for all the rows)
        const size_t key_length_count = input_shapes[InputIdx::KEY_LENGTH].get()[0];
        int64_t longest = 0;
        for (size_t b = 0; b < key_length_count; b++) {
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
    // Every floating-point port (data inputs, state inputs and state outputs) is declared with the
    // precision of Q_IDX, so the kernel sees a single precision (a mismatching edge gets a Reorder).
    // KEY_LENGTH stays i32 -- reordering it to f32 would numerically convert the index values.
    // sel_indices/sel_count are i32; the state outputs never share memory with their inputs
    // (inPlace == -1): the update is out-of-place.
    const auto data_precision = getOriginalInputPrecisionAtPort(InputIdx::Q_IDX);
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        const ov::element::Type precision = i == InputIdx::KEY_LENGTH ? ov::element::i32 : data_precision;
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
        outPortConfigs.emplace_back(LayoutType::ncsp, data_precision, getOutputShapeAtPort(out_port), false, -1);
    }
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref);
}

void QSAIndexer::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto data_precision = getSrcMemoryAtPort(InputIdx::Q_IDX)->getDesc().getPrecision();
    if (data_precision == ov::element::f32) {
        execute_typed<float>();
    } else if (data_precision == ov::element::f16) {
        execute_typed<ov::float16>();
    } else if (data_precision == ov::element::bf16) {
        execute_typed<ov::bfloat16>();
    } else {
        OPENVINO_THROW("QSAIndexer: unsupported precision ", data_precision);
    }
}

template <typename T>
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
    // The RoPE tables may be shared by all the batches ([1, L, Dr]) -- batch 0 is read for every row then.
    const auto rope_cos_batch = [&rope_cos](size_t b) {
        return rope_cos.size(0) == 1 ? 0 : b;
    };
    const auto rope_sin_batch = [&rope_sin](size_t b) {
        return rope_sin.size(0) == 1 ? 0 : b;
    };
    const int64_t L = static_cast<int64_t>(q_idx.size(1));
    const size_t H_iq = q_idx.size(2);
    const size_t D_idx = q_idx.size(3);
    const size_t H_ik = k_idx.size(2);
    const size_t G = H_iq / H_ik;

    const auto* key_length_p = key_length.ptr<int32_t>();
    // key_length is [B] or [1]: one value shared by all the rows
    const size_t key_length_stride = key_length.size(0) == 1 ? 0 : 1;

    // summary_cache is the only state that grows: its output starts as a copy of the blocks of the earlier calls
    // (the input), followed by zeros, and the blocks completed by this call are written over them below.
    const auto summary_cache_in_mem = getSrcMemoryAtPort(InputIdx::SUMMARY_CACHE);
    const auto summary_cache_out_mem = getDstMemoryAtPort(OutputIdx::SUMMARY_CACHE);
    const size_t summary_blocks_in = summary_cache_in_mem->getStaticDims()[2];
    const size_t summary_blocks_out = summary_cache_out_mem->getStaticDims()[2];
    const size_t summary_block_bytes = D_idx * sizeof(T);
    const size_t summary_head_in_bytes = summary_blocks_in * summary_block_bytes;
    const size_t summary_head_out_bytes = summary_blocks_out * summary_block_bytes;
    const size_t summary_copy_bytes = std::min(summary_blocks_in, summary_blocks_out) * summary_block_bytes;
    const auto* summary_cache_src = summary_cache_in_mem->getDataAs<const uint8_t>();
    auto* summary_cache_dst = summary_cache_out_mem->getDataAs<uint8_t>();

    // Every per-row quantity is derived from this call's own key_length/L alone -- no cross-call running counter
    // is kept (see QSAIndexer's class comment).
    struct RowPlan {
        int64_t num_padding;  // left-padded positions of this call's window, skipped everywhere
        int64_t current;      // real tokens of this call
        int64_t tail;         // rows already sitting in the incomplete-block state
        int64_t past_blocks;  // complete blocks of the earlier calls
        int64_t new_blocks;   // blocks this call completes
    };
    const auto plan_row = [&](size_t b) {
        const int64_t key_len = key_length_p[b * key_length_stride];
        const int64_t past_n = std::max<int64_t>(0, key_len - L);
        RowPlan plan{};
        plan.current = key_len - past_n;  // == min(L, key_len)
        plan.num_padding = L - plan.current;
        plan.tail = past_n % r;
        plan.past_blocks = past_n / r;
        plan.new_blocks = (plan.tail + plan.current) / r;
        return plan;
    };

    // One scratch allocation (a second createScratchPadMem() could free the first's memory):
    // a shared header (row plans, head maps, f32 norm weights, a zero row), the f32 mirror of summary_cache when it
    // isn't f32, then one slice per worker thread: floats | (score, block) pairs | chosen block ids.
    constexpr size_t kAlign = 64;
    const size_t max_blocks = summary_blocks_out;  // upper bound of n_complete for every row
    const size_t head_maps_offset = B * sizeof(RowPlan);
    const size_t shared_floats_offset = head_maps_offset + 2 * H_iq * sizeof(size_t);
    const size_t shared_header_bytes = rnd_up(shared_floats_offset + 3 * D_idx * sizeof(float), kAlign);
    // An f32 summary_cache is scored in place; otherwise phase 2b reads the f32 mirror of [B, H_ik, blocks, D_idx].
    constexpr bool summary_is_f32 = std::is_same_v<T, float>;
    const auto data_prec = ov::element::from<T>();
    const size_t summary_mirror_floats = summary_is_f32 ? 0 : B * H_ik * summary_blocks_out * D_idx;
    const size_t shared_bytes = shared_header_bytes + rnd_up(summary_mirror_floats * sizeof(float), kAlign);
    // Float buffers (offsets in floats) of a thread slice. The phases run one after the other, so they share it:
    //   phase 1: pooled (= the first row of q_normed), row_buf, [rope_cos, rope_sin] of the block's reference token
    //   phase 2: q_normed [H_iq, D_idx], row_buf, [rope_cos, rope_sin] of the token
    const size_t q_normed_off = 0;
    const size_t row_buf_off = q_normed_off + H_iq * D_idx;
    const size_t rope_cos_sin_off = row_buf_off + D_idx;  // past_incomplete_rope's [rope_cos, rope_sin] layout
    const size_t thread_floats = rope_cos_sin_off + 2 * rotary_dim;
    using ScoreT = std::pair<float, int32_t>;
    const size_t thread_scores_offset = rnd_up(thread_floats * sizeof(float), kAlign);
    const size_t thread_chosen_offset = rnd_up(thread_scores_offset + max_blocks * sizeof(ScoreT), kAlign);
    const size_t thread_bytes = rnd_up(thread_chosen_offset + block_topk * sizeof(int32_t), kAlign);
    const auto num_threads = static_cast<size_t>(context->getCpuParallel()->get_num_worker_threads());
    auto scratch_desc =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8,
                                               ov::intel_cpu::Shape{shared_bytes + num_threads * thread_bytes});
    const MemoryPtr scratch = context->getScratchPad()->createScratchPadMem(scratch_desc);
    auto* const scratch_base = scratch->getDataAs<uint8_t>();

    auto* const plans = reinterpret_cast<RowPlan*>(scratch_base);
    auto* const kv_of_head = reinterpret_cast<size_t*>(scratch_base + head_maps_offset);
    auto* const sel_of_head = kv_of_head + H_iq;
    auto* const q_norm_weight_f32 = reinterpret_cast<float*>(scratch_base + shared_floats_offset);
    auto* const k_norm_weight_f32 = q_norm_weight_f32 + D_idx;
    auto* const zero_row = k_norm_weight_f32 + D_idx;
    const float* const q_norm_w = row_f32<T>(q_norm_weight, D_idx, q_norm_weight_f32);
    const float* const k_norm_w = row_f32<T>(k_norm_weight, D_idx, k_norm_weight_f32);
    std::fill_n(zero_row, D_idx, 0.0F);
    for (size_t h = 0; h < H_iq; h++) {
        kv_of_head[h] = h / G;
        sel_of_head[h] = (kv_of_head[h] * H_sel) / H_ik;
    }
    auto* const summary_f32_base = summary_is_f32 ? reinterpret_cast<float*>(summary_cache_dst)
                                                  : reinterpret_cast<float*>(scratch_base + shared_header_bytes);
    // f32 view of the summary block `blk` of kv head `h` of row `b`; same layout as the summary_cache output.
    auto summary_f32_row = [&](size_t b, size_t h, size_t blk) {
        return summary_f32_base + ((b * H_ik + h) * summary_blocks_out + blk) * D_idx;
    };

    int64_t max_new_blocks = 0;
    for (size_t b = 0; b < B; b++) {
        plans[b] = plan_row(b);
        OPENVINO_ASSERT(static_cast<size_t>(plans[b].past_blocks + plans[b].new_blocks) <= summary_blocks_out,
                        "QSAIndexer: ",
                        plans[b].past_blocks + plans[b].new_blocks,
                        " blocks do not fit the summary_cache output of ",
                        summary_blocks_out,
                        " blocks");
        max_new_blocks = std::max(max_new_blocks, plans[b].new_blocks);
    }

    // The pending window of a row is [the `tail` rows of the incomplete-block state | the real tokens of this call].
    // Raw K row at pending position `pos`:
    // (a view of the tensor when it is f32, `scratch` otherwise)
    const auto load_pending_row = [&](size_t b, const RowPlan& plan, size_t h, int64_t pos, float* scratch) {
        if (pos < plan.tail) {
            return row_f32<T>(past_incomplete_raw_k_in, D_idx, scratch, b, h, pos);
        }
        return row_f32<T>(k_idx, D_idx, scratch, b, plan.num_padding + pos - plan.tail, h);
    };
    // [rope_cos, rope_sin] of the first token of the block starting at pending position `pos`, in
    // past_incomplete_rope's layout:
    const auto load_block_ref = [&](size_t b, const RowPlan& plan, int64_t pos, float* dst) {
        if (pos < plan.tail) {
            cpu_convert(past_incomplete_rope_in.ptr<T>(b), dst, data_prec, ov::element::f32, 2 * rotary_dim);
        } else {
            const int64_t l = plan.num_padding + pos - plan.tail;
            cpu_convert(rope_cos.ptr<T>(rope_cos_batch(b), l), dst, data_prec, ov::element::f32, rotary_dim);
            cpu_convert(rope_sin.ptr<T>(rope_sin_batch(b), l),
                        dst + rotary_dim,
                        data_prec,
                        ov::element::f32,
                        rotary_dim);
        }
    };
    // This worker thread's slice of the scratch (see the layout comment above).
    const auto thread_slice = [&]() {
        return scratch_base + shared_bytes + static_cast<size_t>(parallel_get_thread_num()) * thread_bytes;
    };

    std::fill_n(sel_indices.ptr<int32_t>(), B * H_sel * static_cast<size_t>(L) * block_topk, static_cast<int32_t>(-1));
    std::fill_n(sel_count.ptr<int32_t>(), B * H_sel * static_cast<size_t>(L), static_cast<int32_t>(0));

    const auto& cpu_parallel = context->getCpuParallel();

    // ---- Phase 0: the summary_cache output starts as the blocks of the earlier calls (+ their f32 mirror) ----
    cpu_parallel->parallel_for2d(B, H_ik, [&](size_t b, size_t h) {
        auto* dst = summary_cache_dst + (b * H_ik + h) * summary_head_out_bytes;
        const auto* src = summary_cache_src + (b * H_ik + h) * summary_head_in_bytes;
        if (summary_copy_bytes != 0) {
            std::memcpy(dst, src, summary_copy_bytes);
        }
        if (summary_head_out_bytes != summary_copy_bytes) {
            std::memset(dst + summary_copy_bytes, 0, summary_head_out_bytes - summary_copy_bytes);
        }
        if constexpr (!summary_is_f32) {
            for (size_t blk = 0; blk < static_cast<size_t>(plans[b].past_blocks); blk++) {
                cpu_convert(summary_cache.ptr<T>(b, h, blk),
                            summary_f32_row(b, h, blk),
                            data_prec,
                            ov::element::f32,
                            D_idx);
            }
        }
    });

    // ---- Phase 1: the blocks this call completes are independent of each other: pool + RMSNorm + RoPE ----
    cpu_parallel->parallel_for3d(B, H_ik, static_cast<size_t>(max_new_blocks), [&](size_t b, size_t h, size_t j) {
        const auto& plan = plans[b];
        if (static_cast<int64_t>(j) >= plan.new_blocks) {
            return;
        }
        auto* const thread_floats_base = reinterpret_cast<float*>(thread_slice());
        float* const row_buf = thread_floats_base + row_buf_off;
        float* const block_ref_cos_sin = thread_floats_base + rope_cos_sin_off;
        float* const pooled = thread_floats_base + q_normed_off;

        const int64_t block_start = static_cast<int64_t>(j) * r;
        std::fill_n(pooled, D_idx, 0.0F);
        for (int64_t p = 0; p < r; p++) {
            const float* const row = load_pending_row(b, plan, h, block_start + p, row_buf);
            for (size_t d = 0; d < D_idx; d++) {
                pooled[d] += row[d];
            }
        }
        const float inv_r = 1.0F / static_cast<float>(r);
        for (size_t d = 0; d < D_idx; d++) {
            pooled[d] *= inv_r;
        }
        load_block_ref(b, plan, block_start, block_ref_cos_sin);
        rmsnorm_partial_rope(pooled,
                             pooled,
                             k_norm_w,
                             D_idx,
                             rotary_dim,
                             eps,
                             block_ref_cos_sin,
                             block_ref_cos_sin + rotary_dim);
        const auto blk = static_cast<size_t>(plan.past_blocks) + j;
        cpu_convert(pooled, summary_cache.ptr<T>(b, h, blk), ov::element::f32, data_prec, D_idx);
        if constexpr (!summary_is_f32) {
            // Phase 2 scores the value as stored (rounded to the cache precision).
            cpu_convert(summary_cache.ptr<T>(b, h, blk),
                        summary_f32_row(b, h, blk),
                        data_prec,
                        ov::element::f32,
                        D_idx);
        }
    });

    // ---- Phase 1b: fully overwrite the fixed-size incomplete-block state ----
    cpu_parallel->parallel_for(B, [&](size_t b) {
        const auto& plan = plans[b];
        auto* const thread_floats_base = reinterpret_cast<float*>(thread_slice());
        float* const row_buf = thread_floats_base + row_buf_off;
        float* const block_ref_cos_sin = thread_floats_base + rope_cos_sin_off;

        const int64_t rest = plan.tail + plan.current - plan.new_blocks * r;
        const int64_t rest_start = plan.new_blocks * r;
        for (int64_t p = 0; p < r - 1; p++) {
            for (size_t h = 0; h < H_ik; h++) {
                const float* src = zero_row;
                if (p < rest) {
                    src = load_pending_row(b, plan, h, rest_start + p, row_buf);
                }
                cpu_convert(src, past_incomplete_raw_k.ptr<T>(b, h, p), ov::element::f32, data_prec, D_idx);
            }
        }
        if (rest > 0) {
            load_block_ref(b, plan, rest_start, block_ref_cos_sin);
        } else {
            std::fill_n(block_ref_cos_sin, 2 * rotary_dim, 0.0F);
        }
        cpu_convert(block_ref_cos_sin, past_incomplete_rope.ptr<T>(b), ov::element::f32, data_prec, 2 * rotary_dim);
    });

    // ---- Phase 2: per real token, query RMSNorm + RoPE, then per-selection-head scoring of the complete blocks ----
    cpu_parallel->parallel_for2d(B, static_cast<size_t>(L), [&](size_t b, size_t l_u) {
        const auto& plan = plans[b];
        const auto l = static_cast<int64_t>(l_u);
        if (l < plan.num_padding) {
            return;
        }
        auto* const thread_base = thread_slice();
        auto* const thread_floats_base = reinterpret_cast<float*>(thread_base);
        float* const q_normed = thread_floats_base + q_normed_off;  // [H_iq, D_idx]
        float* const row_buf = thread_floats_base + row_buf_off;
        float* const rope_cos_buf = thread_floats_base + rope_cos_sin_off;
        float* const rope_sin_buf = rope_cos_buf + rotary_dim;
        auto* const scores = reinterpret_cast<ScoreT*>(thread_base + thread_scores_offset);
        auto* const chosen = reinterpret_cast<int32_t*>(thread_base + thread_chosen_offset);

        // ---- Phase 2a: query RMSNorm + partial RoPE (per index-query head) ----
        const float* const cos_row = row_f32<T>(rope_cos, rotary_dim, rope_cos_buf, rope_cos_batch(b), l);
        const float* const sin_row = row_f32<T>(rope_sin, rotary_dim, rope_sin_buf, rope_sin_batch(b), l);
        for (size_t h = 0; h < H_iq; h++) {
            const float* const q_row = row_f32<T>(q_idx, D_idx, row_buf, b, l, h);
            rmsnorm_partial_rope(q_row, q_normed + h * D_idx, q_norm_w, D_idx, rotary_dim, eps, cos_row, sin_row);
        }

        // ---- Phase 2b: per-selection-head complete-block scoring + top-k ----
        // The complete blocks visible to this token: the earlier calls' plus those finished up to this token.
        const int64_t v = l - plan.num_padding;
        const int64_t n_complete = plan.past_blocks + (plan.tail + v + 1) / r;
        for (size_t sh = 0; sh < H_sel; sh++) {
            size_t k = 0;
            if (n_complete > 0) {
                for (int64_t bidx = 0; bidx < n_complete; bidx++) {
                    float score = 0.0F;
                    for (size_t h = 0; h < H_iq; h++) {
                        if (sel_of_head[h] != sh) {
                            continue;
                        }
                        const float* const block_f32 = summary_f32_row(b, kv_of_head[h], static_cast<size_t>(bidx));
                        float dot = 0.0F;
                        for (size_t d = 0; d < D_idx; d++) {
                            dot += q_normed[h * D_idx + d] * block_f32[d];
                        }
                        score += std::max(0.0F, dot);
                    }
                    scores[static_cast<size_t>(bidx)] = {score * scale, static_cast<int32_t>(bidx)};
                }
                k = std::min(block_topk, static_cast<size_t>(n_complete));
                std::partial_sort(scores, scores + k, scores + n_complete, [](const auto& a, const auto& b2) {
                    if (a.first != b2.first) {
                        return a.first > b2.first;
                    }
                    return a.second < b2.second;
                });
                for (size_t i = 0; i < k; i++) {
                    chosen[i] = scores[i].second;
                }
                std::sort(chosen, chosen + k);
            }

            auto* out_row = sel_indices.ptr<int32_t>(b, sh, l_u, 0);
            for (size_t i = 0; i < k; i++) {
                out_row[i] = static_cast<int32_t>(chosen[i]);
            }
            *sel_count.ptr<int32_t>(b, sh, l_u) = static_cast<int32_t>(k);
        }
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
        // All floating-point ports share one precision (the fusion pass converts the graph to that).
        const auto data_precision = op->get_input_element_type(InputIdx::Q_IDX);
        if (data_precision != ov::element::f32 && data_precision != ov::element::f16 &&
            data_precision != ov::element::bf16) {
            errorMessage = "QSAIndexer CPU implementation only supports f32, f16, or bf16 data inputs.";
            return false;
        }
        for (auto port : float_ports) {
            if (op->get_input_element_type(port) != data_precision) {
                errorMessage = "QSAIndexer CPU implementation requires all floating-point inputs to have the same "
                               "precision.";
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
