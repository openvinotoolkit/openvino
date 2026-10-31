// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_qsa_indexer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "graph_context.h"
#include "kernels/scaled_attn/common.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "qsa_indexer_common.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include "kernels/x64/qsa_brgemm_scorer.hpp"
#endif

namespace ov::intel_cpu::node {

namespace {

using InputIdx = ov::op::internal::PagedQSAIndexer::InputIdx;

/// @brief Fixed output order of ov::op::internal::PagedQSAIndexer (not part of the op's public API).
struct OutputIdx {
    static constexpr size_t SEL_INDICES = 0;  ///< [tokens, h_sel, block_topk]
    static constexpr size_t SEL_COUNT = 1;    ///< [tokens, h_sel]
};

using qsa::rmsnorm_partial_rope;

// All floating-point data ports of PagedQSAIndexer (index_query, raw_index_keys, rope_cos, rope_sin,
// q_norm_weight, k_norm_weight, index_key_cache, index_block_rope_cache, block_index_keys_cache) may
// independently be negotiated as f32, f16, or bf16 -- these convert only the elements actually
// touched, so compute stays fp32 without paying for a whole-tensor Reorder on every call over
// caches that grow every step. Every port has its own precision, so unlike qsa_indexer.cpp (one type T) the
// precision is looked up per call.
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

// Read-only f32 view of a row of a tensor of any of the three float types: the row itself when the tensor is f32 (no
// copy), otherwise the row converted into `scratch`.
template <typename... Idx>
const float* row_view(const PlainTensor& t, size_t len, float* scratch, Idx... idx) {
    if (t.get_precision() == ov::element::f32) {
        return t.ptr<float>(idx...);
    }
    load_row_f32(t, len, scratch, idx...);
    return scratch;
}

// What a call knows about one sequence: where its tokens and its page-table row are, and which index blocks it
// completes. past_blocks = past_len / r blocks were complete before the call, new_blocks = (past_len + q_len) / r -
// past_blocks are completed by it.
struct SeqPlan {
    int64_t token_begin;
    int64_t q_len;
    int64_t past_len;
    int64_t block_begin;     // first entry of the sequence in block_indices
    int64_t block_capacity;  // index blocks that fit the sequence's pages
    int64_t past_blocks;
    int64_t new_blocks;
};

// Float buffers (offsets in floats) and the (score, block) / chosen arrays (offsets in bytes) of a worker thread's
// scratch slice; the phases run one after the other, so they share the buffers.
struct ThreadLayout {
    size_t q_normed_off = 0;  // [H_iq, D_idx] normalized + rotated index queries
    size_t row_buf_off;       // [D_idx] a converted row
    size_t summary_buf_off;   // [D_idx] a converted / pooled summary row
    size_t pooled_off;        // [D_idx] the mean of the raw keys of a block
    size_t cos_sin_off;       // [2 * rotary_dim] the interleaved cos / sin row of a block
    size_t cos_tmp_off;       // [rotary_dim] a converted cos row
    size_t sin_tmp_off;       // [rotary_dim] a converted sin row
    size_t scores_off;        // bytes: (score, block) pairs of the complete blocks
    size_t chosen_off;        // bytes: block ids of the selection
    size_t bytes;

    static constexpr size_t kAlign = 64;

    ThreadLayout(size_t H_iq, size_t D_idx, size_t rotary_dim, size_t max_blocks, size_t block_topk)
        : row_buf_off(q_normed_off + H_iq * D_idx),
          summary_buf_off(row_buf_off + D_idx),
          pooled_off(summary_buf_off + D_idx),
          cos_sin_off(pooled_off + D_idx),
          cos_tmp_off(cos_sin_off + 2 * rotary_dim),
          sin_tmp_off(cos_tmp_off + rotary_dim),
          scores_off(rnd_up((sin_tmp_off + rotary_dim) * sizeof(float), kAlign)),
          chosen_off(rnd_up(scores_off + max_blocks * sizeof(std::pair<float, int32_t>), kAlign)),
          bytes(rnd_up(chosen_off + block_topk * sizeof(int32_t), kAlign)) {}
};

// Everything the three phases read, see PagedQSAIndexer::execute_typed().
struct IndexerArgs {
    const PlainTensor& q_idx;
    const PlainTensor& k_idx;
    const PlainTensor& rope_cos;
    const PlainTensor& rope_sin;
    const PlainTensor& block_indices;
    const PlainTensor& raw_k_cache;
    const PlainTensor& indexer_rope_cache;
    const PlainTensor& summary_cache;
    const PlainTensor& sel_indices;
    const PlainTensor& sel_count;
    int64_t r;  // key_block_size
    int64_t page_size;
    int64_t slots_per_page;
    size_t rotary_dim;
    size_t D_idx;
    size_t H_iq;
    size_t H_ik;
    size_t H_sel;
    size_t block_topk;
    float eps;
    float scale;
    const SeqPlan* plans;
    const int32_t* token_seq;  // per token: its sequence, -1 for a token that belongs to none
    const size_t* sel_of_kv;   // per index-key head: its selection head
    const float* q_norm_w;
    const float* k_norm_w;
    const ThreadLayout& thread_layout;
    uint8_t* thread_slices;

    // Maps a logical token position (relative to the sequence's own 0..S history) to the physical (block, slot) address
    // into INDEX_KEY_CACHE through the shared page table.
    [[nodiscard]] std::pair<size_t, size_t> token_phys(const SeqPlan& plan, int64_t logical_pos) const {
        const int64_t phys_idx = plan.block_begin + logical_pos / page_size;
        const auto phys_block = static_cast<size_t>(*block_indices.ptr<int32_t>(static_cast<size_t>(phys_idx)));
        return {phys_block, static_cast<size_t>(logical_pos % page_size)};
    }
    // Maps a (sequence-local, 0-based) completed-block index to the physical (block, slot) address into
    // BLOCK_INDEX_KEYS_CACHE / INDEX_BLOCK_ROPE_CACHE.
    [[nodiscard]] std::pair<size_t, size_t> block_phys(const SeqPlan& plan, int64_t block_idx) const {
        const int64_t phys_idx = plan.block_begin + (block_idx * r) / page_size;
        const auto phys_block = static_cast<size_t>(*block_indices.ptr<int32_t>(static_cast<size_t>(phys_idx)));
        return {phys_block, static_cast<size_t>(block_idx % slots_per_page)};
    }
    [[nodiscard]] uint8_t* thread_slice() const {
        return thread_slices + static_cast<size_t>(parallel_get_thread_num()) * thread_layout.bytes;
    }
};

#ifdef OPENVINO_ARCH_X86_64
// The paged layout of a call as the brgemm scorer sees it (see BrgemmBlockScorer): the rows are the sequences, the
// token v of a sequence is the flat token plan.token_begin + v, and the summary blocks of a sequence are addressed
// through the page table: a page holds `slots_per_page` consecutive blocks of a K head, and the pages of a sequence
// are anywhere in the cache. `T` is the precision of the summary cache.
template <typename T>
class PagedSource {
public:
    using Element = T;

    explicit PagedSource(const IndexerArgs& args) : m_a(args) {}

    [[nodiscard]] int64_t tokens(size_t seq) const {
        return m_a.plans[seq].q_len;
    }
    // the complete blocks up to the one the token completes
    [[nodiscard]] int64_t blocks_visible(size_t seq, int64_t v) const {
        return (m_a.plans[seq].past_len + v + 1) / m_a.r;
    }
    [[nodiscard]] int64_t blocks(size_t seq) const {
        return m_a.plans[seq].past_blocks + m_a.plans[seq].new_blocks;
    }
    [[nodiscard]] const float* query_row(size_t seq, int64_t v, size_t h, float* buf) const {
        return row_view(m_a.q_idx, m_a.D_idx, buf, token(seq, v), h * m_a.D_idx);
    }
    [[nodiscard]] const float* rope_cos(size_t seq, int64_t v, float* buf) const {
        return row_view(m_a.rope_cos, m_a.rotary_dim, buf, token(seq, v));
    }
    [[nodiscard]] const float* rope_sin(size_t seq, int64_t v, float* buf) const {
        return row_view(m_a.rope_sin, m_a.rotary_dim, buf, token(seq, v));
    }
    [[nodiscard]] const float* q_norm_weight() const {
        return m_a.q_norm_w;
    }
    // 32 blocks that lie in one page are used in place; blocks of several pages (a small page, or the end of the
    // sequence) are gathered, run by run, into the zero-padded stage.
    [[nodiscard]] const T* block_rows(size_t seq, size_t hk, size_t first, size_t count, T* stage) const {
        const auto& plan = m_a.plans[seq];
        const auto slots = static_cast<size_t>(m_a.slots_per_page);
        const auto [first_page, first_slot] = m_a.block_phys(plan, static_cast<int64_t>(first));
        if (count == qsa::kScoreBlockRows && first_slot + count <= slots) {
            return m_a.summary_cache.template ptr<T>(first_page, hk, first_slot);
        }
        const size_t dim = m_a.D_idx;
        size_t done = 0;
        while (done < count) {
            const auto [page, slot] = m_a.block_phys(plan, static_cast<int64_t>(first + done));
            const size_t run = std::min(count - done, slots - slot);
            std::copy_n(m_a.summary_cache.template ptr<T>(page, hk, slot), run * dim, stage + done * dim);
            done += run;
        }
        std::fill_n(stage + count * dim, (qsa::kScoreBlockRows - count) * dim, T{});
        return stage;
    }
    [[nodiscard]] int32_t* selection_indices(size_t seq, int64_t v, size_t sh) const {
        return m_a.sel_indices.template ptr<int32_t>(token(seq, v), sh, 0);
    }
    [[nodiscard]] int32_t* selection_count(size_t seq, int64_t v, size_t sh) const {
        return m_a.sel_count.template ptr<int32_t>(token(seq, v), sh);
    }

private:
    [[nodiscard]] size_t token(size_t seq, int64_t v) const {
        return static_cast<size_t>(m_a.plans[seq].token_begin + v);
    }

    const IndexerArgs& m_a;
};
#endif

// Phase 1, per token: append the token's raw index keys to the paged cache, and when it starts a block record the
// interleaved cos/sin row of that first token, so the block's summary (finalized once the block completes, possibly in
// a later call) can be RoPE'ed with the exact coefficients of its first token, regardless of KV-cache slot vs logical
// position drift (continuous batching, chunked prefill, mRoPE, ...). The capacity bound skips a trailing block that
// starts within the allocated page range but has no room to ever complete.
void store_token_rows(const IndexerArgs& a, size_t token) {
    const auto& plan = a.plans[a.token_seq[token]];
    const int64_t logical_pos = plan.past_len + (static_cast<int64_t>(token) - plan.token_begin);
    auto* const floats = reinterpret_cast<float*>(a.thread_slice());
    const auto& lay = a.thread_layout;
    float* const row_buf = floats + lay.row_buf_off;

    if (logical_pos % a.r == 0) {
        const int64_t block_idx = logical_pos / a.r;
        if (block_idx < plan.block_capacity) {
            float* const cos_sin = floats + lay.cos_sin_off;
            const float* const cos_row = row_view(a.rope_cos, a.rotary_dim, floats + lay.cos_tmp_off, token);
            const float* const sin_row = row_view(a.rope_sin, a.rotary_dim, floats + lay.sin_tmp_off, token);
            for (size_t j = 0; j < a.rotary_dim; j++) {
                cos_sin[2 * j] = cos_row[j];
                cos_sin[2 * j + 1] = sin_row[j];
            }
            const auto [phys_block, slot] = a.block_phys(plan, block_idx);
            store_row_f32(a.indexer_rope_cache, 2 * a.rotary_dim, cos_sin, phys_block, slot);
        }
    }
    const auto [cur_phys_block, cur_slot] = a.token_phys(plan, logical_pos);
    for (size_t h = 0; h < a.H_ik; h++) {
        const float* const row = row_view(a.k_idx, a.D_idx, row_buf, token, h * a.D_idx);
        store_row_f32(a.raw_k_cache, a.D_idx, row, cur_phys_block, h, cur_slot);
    }
}

// Phase 2, per (sequence, index-key head, block completed by the call): the summary of the block is the mean of its raw
// keys, RMSNorm'ed and RoPE'd with the coefficients of its first token. The blocks are independent of each other.
void pool_block(const IndexerArgs& a, size_t seq, size_t h, size_t j) {
    const auto& plan = a.plans[seq];
    auto* const floats = reinterpret_cast<float*>(a.thread_slice());
    const auto& lay = a.thread_layout;
    float* const row_buf = floats + lay.row_buf_off;
    float* const pooled = floats + lay.pooled_off;
    float* const summary_buf = floats + lay.summary_buf_off;
    float* const cos_sin_buf = floats + lay.cos_sin_off;

    const int64_t block_idx = plan.past_blocks + static_cast<int64_t>(j);
    const int64_t block_start = block_idx * a.r;
    const auto [block_phys_block, block_slot] = a.block_phys(plan, block_idx);
    std::fill_n(pooled, a.D_idx, 0.0F);
    for (int64_t p = block_start; p < block_start + a.r; p++) {
        const auto [p_phys_block, p_slot] = a.token_phys(plan, p);
        const float* const row = row_view(a.raw_k_cache, a.D_idx, row_buf, p_phys_block, h, p_slot);
        for (size_t d = 0; d < a.D_idx; d++) {
            pooled[d] += row[d];
        }
    }
    const float inv_r = 1.0F / static_cast<float>(a.r);
    for (size_t d = 0; d < a.D_idx; d++) {
        pooled[d] *= inv_r;
    }
    const float* const cos_sin =
        row_view(a.indexer_rope_cache, 2 * a.rotary_dim, cos_sin_buf, block_phys_block, block_slot);
    rmsnorm_partial_rope(pooled, summary_buf, a.k_norm_w, a.D_idx, a.rotary_dim, a.eps, cos_sin, cos_sin + 1, 2);
    store_row_f32(a.summary_cache, a.D_idx, summary_buf, block_phys_block, h, block_slot);
}

// Phase 3, per token: the query RMSNorm + partial RoPE of every index-query head, then per selection head the score of
// every complete block (the sum over the heads of the selection head of max(0, q . summary)) and the top-k blocks in
// ascending block order. A summary row is loaded (converted) once per block and index-key head, whatever the number of
// query heads that share it.
void select_blocks(const IndexerArgs& a, size_t token) {
    const auto& plan = a.plans[a.token_seq[token]];
    const int64_t logical_pos = plan.past_len + (static_cast<int64_t>(token) - plan.token_begin);
    auto* const thread_base = a.thread_slice();
    auto* const floats = reinterpret_cast<float*>(thread_base);
    const auto& lay = a.thread_layout;
    float* const q_normed = floats + lay.q_normed_off;
    float* const row_buf = floats + lay.row_buf_off;
    float* const summary_buf = floats + lay.summary_buf_off;
    auto* const scores = reinterpret_cast<std::pair<float, int32_t>*>(thread_base + lay.scores_off);
    auto* const chosen = reinterpret_cast<int32_t*>(thread_base + lay.chosen_off);

    const float* const cos_row = row_view(a.rope_cos, a.rotary_dim, floats + lay.cos_tmp_off, token);
    const float* const sin_row = row_view(a.rope_sin, a.rotary_dim, floats + lay.sin_tmp_off, token);
    for (size_t h = 0; h < a.H_iq; h++) {
        const float* const q_row = row_view(a.q_idx, a.D_idx, row_buf, token, h * a.D_idx);
        rmsnorm_partial_rope(q_row, q_normed + h * a.D_idx, a.q_norm_w, a.D_idx, a.rotary_dim, a.eps, cos_row, sin_row);
    }

    // The complete blocks visible to this token, including the one this token completes.
    const int64_t n_complete = (logical_pos + 1) / a.r;
    const size_t group = a.H_iq / a.H_ik;  // query heads per index-key head
    for (size_t sh = 0; sh < a.H_sel; sh++) {
        size_t k = 0;
        if (n_complete > 0) {
            for (int64_t bidx = 0; bidx < n_complete; bidx++) {
                const auto [s_phys_block, s_slot] = a.block_phys(plan, bidx);
                float score = 0.0F;
                for (size_t hk = 0; hk < a.H_ik; hk++) {
                    if (a.sel_of_kv[hk] != sh) {
                        continue;
                    }
                    const float* const block_row =
                        row_view(a.summary_cache, a.D_idx, summary_buf, s_phys_block, hk, s_slot);
                    for (size_t h = hk * group; h < (hk + 1) * group; h++) {
                        float dot = 0.0F;
                        for (size_t d = 0; d < a.D_idx; d++) {
                            dot += q_normed[h * a.D_idx + d] * block_row[d];
                        }
                        score += std::max(0.0F, dot);
                    }
                }
                scores[static_cast<size_t>(bidx)] = {score * a.scale, static_cast<int32_t>(bidx)};
            }
            k = std::min(a.block_topk, static_cast<size_t>(n_complete));
            std::partial_sort(scores, scores + k, scores + n_complete, [](const auto& x, const auto& y) {
                if (x.first != y.first) {
                    return x.first > y.first;
                }
                return x.second < y.second;
            });
            for (size_t i = 0; i < k; i++) {
                chosen[i] = scores[i].second;
            }
            std::sort(chosen, chosen + k);
        }

        auto* out_row = a.sel_indices.ptr<int32_t>(token, sh, 0);
        for (size_t i = 0; i < k; i++) {
            out_row[i] = chosen[i];
        }
        *a.sel_count.ptr<int32_t>(token, sh) = static_cast<int32_t>(k);
    }
}

}  // namespace

#ifdef OPENVINO_ARCH_X86_64
// Query heads per K head if the scoring of this node runs on brgemm, 0 if not. The query is rounded to the precision of
// the summary blocks for the brgemm products, which is only done when it is that precision already.
static size_t brgemm_group_size(const Node& node,
                                const ov::element::Type& query_precision,
                                const ov::element::Type& summary_precision,
                                size_t selection_heads,
                                size_t dim) {
    const auto q_width = node.getInputShapeAtPort(InputIdx::INDEX_QUERY).getDims()[1];
    const auto k_width = node.getInputShapeAtPort(InputIdx::RAW_INDEX_KEYS).getDims()[1];
    if (query_precision != summary_precision || q_width == Shape::UNDEFINED_DIM || k_width == Shape::UNDEFINED_DIM) {
        return 0;
    }
    return qsa::brgemm_score_group_size(summary_precision, q_width / dim, k_width / dim, selection_heads, dim);
}
#endif

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
    // The scoring runs on brgemm where the CPU, the precisions and the head counts allow it (see createPrimitive()).
    impl_desc_type impl = impl_desc_type::ref;
#ifdef OPENVINO_ARCH_X86_64
    const auto summary_precision = getOriginalInputPrecisionAtPort(InputIdx::BLOCK_INDEX_KEYS_CACHE);
    if (brgemm_group_size(*this,
                          getOriginalInputPrecisionAtPort(InputIdx::INDEX_QUERY),
                          summary_precision,
                          m_config.num_selection_heads,
                          m_config.indexer_head_dim) != 0) {
        impl = qsa::score_impl_type(summary_precision);
    }
#endif
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl);
}

void PagedQSAIndexer::createPrimitive() {
#ifdef OPENVINO_ARCH_X86_64
    // The scorer depends on the configuration, the static head counts and the precision only: like the executor of the
    // ScaledDotProductAttention it comes from the params cache once here, never while the model runs.
    // initSupportedPrimitiveDescriptors() registered the brgemm implementation type for the nodes that can have it.
    const auto& config = getSelectedPrimitiveDescriptor()->getConfig();
    const auto precision = config.inConfs[InputIdx::BLOCK_INDEX_KEYS_CACHE].getMemDesc()->getPrecision();
    const size_t dim = m_config.indexer_head_dim;
    const auto q_width = getInputShapeAtPort(InputIdx::INDEX_QUERY).getDims()[1];
    const auto k_width = getInputShapeAtPort(InputIdx::RAW_INDEX_KEYS).getDims()[1];
    // the query is rounded to the precision of the summary blocks for the brgemm products: only done when it is that
    // precision already
    if (config.inConfs[InputIdx::INDEX_QUERY].getMemDesc()->getPrecision() == precision &&
        q_width != Shape::UNDEFINED_DIM && k_width != Shape::UNDEFINED_DIM) {
        const qsa::BrgemmBlockScorer::Config scorer_config{
            q_width / dim,
            k_width / dim,
            m_config.num_selection_heads,
            dim,
            m_config.indexer_rotary_dim,
            m_config.block_topk,
            static_cast<size_t>(context->getCpuParallel()->get_num_worker_threads()),
            m_config.epsilon,
            ov::op::internal::PagedQSAIndexer::effective_scale(m_config),
            precision};
        m_scorer = qsa::BrgemmBlockScorer::get(context->getParamsCache(), scorer_config);
    }
    if (!m_scorer) {
        // nothing to score with on brgemm: the scalar scoring runs, so the node is a reference implementation
        getSelectedPrimitiveDescriptor()->setImplementationType(impl_desc_type::ref);
    }
#endif
}

void PagedQSAIndexer::execute([[maybe_unused]] const dnnl::stream& strm) {
    // The precision of the summary blocks: the one the block scoring on brgemm works in.
    const auto summary_precision = getSrcMemoryAtPort(InputIdx::BLOCK_INDEX_KEYS_CACHE)->getDesc().getPrecision();
    if (summary_precision == ov::element::f32) {
        execute_typed<float>();
    } else if (summary_precision == ov::element::f16) {
        execute_typed<ov::float16>();
    } else {
        execute_typed<ov::bfloat16>();
    }
}

template <typename T>
void PagedQSAIndexer::execute_typed() {
    const auto& c = m_config;
    const auto r = static_cast<int64_t>(c.key_block_size);
    const size_t rotary_dim = c.indexer_rotary_dim;
    const size_t D_idx = c.indexer_head_dim;
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

    const auto* subsequence_begins_p = subsequence_begins.ptr<int32_t>();
    const auto* block_indices_begins_p = block_indices_begins.ptr<int32_t>();
    const auto* past_lens_p = past_lens.ptr<int32_t>();

    const auto seq_tensor_size = subsequence_begins.size(0);
    OPENVINO_ASSERT(seq_tensor_size >= 1, "[CPU] paged_qsa_indexer: subsequence_begins must be non-empty");
    const size_t num_sequences = seq_tensor_size - 1;

    // The plan of every sequence (see SeqPlan) and the sequence of every token. The scores of a token are for at most
    // max_blocks complete blocks, the pooling of a call is for at most max_new_blocks blocks of a sequence.
    std::vector<SeqPlan> plans(num_sequences);
    int64_t max_blocks = 0;
    int64_t max_new_blocks = 0;
    int64_t max_q_len = 0;
    for (size_t seq = 0; seq < num_sequences; seq++) {
        const int64_t block_begin = block_indices_begins_p[seq];
        const int64_t seq_blocks = std::max<int64_t>(block_indices_begins_p[seq + 1] - block_begin, 0);
        OPENVINO_ASSERT(seq_blocks > 0, "[CPU] paged_qsa_indexer: each sequence must have at least one cache block");
        SeqPlan plan{};
        plan.token_begin = subsequence_begins_p[seq];
        plan.q_len = std::max<int64_t>(subsequence_begins_p[seq + 1] - plan.token_begin, 0);
        plan.past_len = past_lens_p[seq];
        plan.block_begin = block_begin;
        plan.block_capacity = seq_blocks * slots_per_page;
        plan.past_blocks = plan.past_len / r;
        plan.new_blocks = (plan.past_len + plan.q_len) / r - plan.past_blocks;
        OPENVINO_ASSERT(plan.past_blocks + plan.new_blocks <= plan.block_capacity,
                        "[CPU] paged_qsa_indexer: the completed blocks do not fit the sequence's cache blocks");
        max_blocks = std::max(max_blocks, plan.past_blocks + plan.new_blocks);
        max_new_blocks = std::max(max_new_blocks, plan.new_blocks);
        max_q_len = std::max(max_q_len, plan.q_len);
        plans[seq] = plan;
    }

    // Scoring on brgemm (see qsa_brgemm_scorer.hpp; the scorer was taken from the params cache in createPrimitive(),
    // the call only plans its sizes, which have to be known for the allocation below).
    const auto num_threads = static_cast<size_t>(context->getCpuParallel()->get_num_worker_threads());
#ifdef OPENVINO_ARCH_X86_64
    const bool use_scorer = m_scorer && max_blocks > 0;
    qsa::BrgemmBlockScorer::Plan scorer_plan;
    if (use_scorer) {
        scorer_plan = m_scorer->plan(num_sequences, max_q_len, max_blocks);
    }
    const size_t scorer_shared_bytes = scorer_plan.shared_bytes;
    const size_t scorer_thread_bytes = scorer_plan.thread_bytes;
#else
    constexpr size_t scorer_shared_bytes = 0;
    constexpr size_t scorer_thread_bytes = 0;
#endif

    // One scratch allocation (a second createScratchPadMem() could free the first's memory): a header (token ->
    // sequence array, the head maps and the f32 norm weights), what the brgemm scorer shares between the threads, then
    // one slice per worker thread, see ThreadLayout (the scorer lays out its own slices over the same space, after
    // phases 1 and 2 have used theirs).
    constexpr size_t kAlign = 64;
    const size_t token_seq_off = 0;
    const size_t sel_of_kv_off = rnd_up(token_seq_off + tokens * sizeof(int32_t), kAlign);
    const size_t weights_off = rnd_up(sel_of_kv_off + H_ik * sizeof(size_t), kAlign);
    const size_t header_bytes = rnd_up(weights_off + 2 * D_idx * sizeof(float), kAlign);
    const ThreadLayout thread_layout(H_iq, D_idx, rotary_dim, static_cast<size_t>(max_blocks), block_topk);
    const size_t scorer_shared_offset = header_bytes;
    const size_t threads_offset = rnd_up(scorer_shared_offset + scorer_shared_bytes, kAlign);
    const size_t thread_slice_bytes = std::max(thread_layout.bytes, scorer_thread_bytes);
    const MemoryPtr scratch = context->getScratchPad()->createScratchPadMem(std::make_shared<CpuBlockedMemoryDesc>(
        ov::element::u8,
        ov::intel_cpu::Shape{threads_offset + num_threads * thread_slice_bytes}));
    auto* const scratch_base = scratch->getDataAs<uint8_t>();

    auto* const token_seq = reinterpret_cast<int32_t*>(scratch_base + token_seq_off);
    auto* const sel_of_kv = reinterpret_cast<size_t*>(scratch_base + sel_of_kv_off);
    auto* const q_norm_w = reinterpret_cast<float*>(scratch_base + weights_off);
    auto* const k_norm_w = q_norm_w + D_idx;
    std::fill_n(token_seq, tokens, static_cast<int32_t>(-1));
    for (size_t seq = 0; seq < num_sequences; seq++) {
        for (int64_t t = plans[seq].token_begin; t < plans[seq].token_begin + plans[seq].q_len; t++) {
            if (static_cast<size_t>(t) < tokens) {
                token_seq[t] = static_cast<int32_t>(seq);
            }
        }
    }
    for (size_t hk = 0; hk < H_ik; hk++) {
        sel_of_kv[hk] = (hk * H_sel) / H_ik;
    }
    load_row_f32(q_norm_weight, D_idx, q_norm_w);
    load_row_f32(k_norm_weight, D_idx, k_norm_w);

    std::fill_n(sel_indices.ptr<int32_t>(), tokens * H_sel * block_topk, static_cast<int32_t>(-1));
    std::fill_n(sel_count.ptr<int32_t>(), tokens * H_sel, static_cast<int32_t>(0));

    const IndexerArgs args{q_idx,
                           k_idx,
                           rope_cos,
                           rope_sin,
                           block_indices,
                           raw_k_cache,
                           indexer_rope_cache,
                           summary_cache,
                           sel_indices,
                           sel_count,
                           r,
                           page_size,
                           slots_per_page,
                           rotary_dim,
                           D_idx,
                           H_iq,
                           H_ik,
                           H_sel,
                           block_topk,
                           c.epsilon,
                           ov::op::internal::PagedQSAIndexer::effective_scale(c),
                           plans.data(),
                           token_seq,
                           sel_of_kv,
                           q_norm_w,
                           k_norm_w,
                           thread_layout,
                           scratch_base + threads_offset};

    const auto& cpu_parallel = context->getCpuParallel();
    // Phase 1: the raw keys of the tokens (and the rope rows of the blocks they start) go to the caches; disjoint
    // slots.
    cpu_parallel->parallel_for(tokens, [&](size_t token) {
        if (token_seq[token] >= 0) {
            store_token_rows(args, token);
        }
    });
    // Phase 2: the blocks this call completes are independent of each other (they read the raw rows of this and of
    // earlier calls, written by phase 1 or before).
    if (max_new_blocks > 0) {
        cpu_parallel->parallel_for3d(num_sequences,
                                     H_ik,
                                     static_cast<size_t>(max_new_blocks),
                                     [&](size_t seq, size_t h, size_t j) {
                                         if (static_cast<int64_t>(j) < plans[seq].new_blocks) {
                                             pool_block(args, seq, h, j);
                                         }
                                     });
    }
    // Phase 3: a token scores the complete blocks up to the one it completes, all written by now.
#ifdef OPENVINO_ARCH_X86_64
    if (use_scorer) {
        const PagedSource<T> source(args);
        m_scorer->run(scorer_plan,
                      source,
                      scratch_base + scorer_shared_offset,
                      scratch_base + threads_offset,
                      *cpu_parallel);
        return;
    }
#endif
    cpu_parallel->parallel_for(tokens, [&](size_t token) {
        if (token_seq[token] >= 0) {
            select_blocks(args, token);
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
