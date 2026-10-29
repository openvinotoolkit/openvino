// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <memory>
#include <utility>

#include "../utils/kernel_generator.hpp"
#include "paged_attention_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

// sdpa_micro reads the per-token token_type_ids group bounds precomputed by the pa_tt_bounds stage
// (internal buffer 4, right after the micro-kernel buffer 3). Not used with scores output or adaptive RKV,
// which add internal buffers before that index.
inline bool pa_micro_uses_tt_bounds(const paged_attention& desc) {
    return desc.has_token_type_ids && !desc.has_scores_output() && !desc.has_adaptive_rkv;
}

enum class PagedAttentionStage : uint8_t { GENERATE = 0, PREFILL = 1, MIXED = 2, UNKNOWN = 3 };

struct PagedAttentionRuntimeParams : public ImplRuntimeParams {
    PagedAttentionStage stage;
    size_t num_of_partitions;
    size_t partition_size;
    size_t max_context_len;
    size_t paged_attention_aligned_seq_len;
    size_t sdpa_opt_seq_len_partition_size;

    size_t paged_attention_snap_kv_tokens;
    bool use_micro_sdpa = false;
    bool use_gqa_kernel = false;
    size_t query_block_size = 16;
    // Split-K for the MIXED-stage micro SDPA kernels: number of KV partitions (1 = no split).
    size_t micro_splitk_partitions = 1;
    // Partition alignment (K tile of the micro kernel); partition sizes are multiples of it.
    size_t micro_splitk_align = 16;
    // MIXED stage runs the GQA-packed micro SDPA kernel (see pa_micro_gqa_packed_supported).
    bool micro_gqa_packed = false;
};

// Split-K (KV partitioning) for the MIXED-stage micro SDPA kernels. It is applied to calls with small
// per-sequence q_len (speculative decoding verify / draft steps) over long contexts, where the regular
// grid (query blocks x heads) is too small to occupy the GPU.
// Target KV partition size in tokens.
constexpr size_t pa_micro_splitk_size = 1024;
// Split-K is applied only if every subsequence has q_len <= this value.
constexpr size_t pa_micro_splitk_max_q = 16;
// Split-K is applied only if max_context_len >= this value.
constexpr size_t pa_micro_splitk_min_ctx = 4096;
// Internal buffer indices of the split-K partial results in the PA MIXED micro layout. These are the
// same slots pa_micro_uses_tt_bounds() puts tt_bounds in; the two are mutually exclusive (see below).
enum class PaMicroSplitKBuffer : size_t { STATS = 4, TMP_OUT = 5 };
// Placeholder size of those buffers when split-K is off; the kernel never dereferences them.
constexpr size_t pa_micro_splitk_placeholder_bytes = 16;

// Split-K is only supported for the plain MIXED micro path (no sinks, scores output, sliding window,
// adaptive RKV or token_type_ids, which would require extra handling or shift internal buffer indices).
// token_type_ids is excluded on both counts: its group bounds narrow the K window that the split-K
// partitions divide (window_k_begin in sdpa_micro.cl), and pa_tt_bounds claims internal buffer 4, which
// splitk_stats/splitk_tmp_out occupy. Bidirectional-masked calls keep the unsplit MIXED micro kernel.
inline bool pa_micro_splitk_supported(const paged_attention& desc) {
    return !desc.has_sink_input && !desc.has_scores_output() && desc.sliding_window == 0 && !desc.has_adaptive_rkv && !desc.has_token_type_ids;
}

// Partial outputs are kept as half in the temporary buffer -> f16 query/output only.
inline bool pa_micro_splitk_supported(const kernel_impl_params& params) {
    const auto desc = params.typed_desc<paged_attention>();
    return pa_micro_splitk_supported(*desc) && params.input_layouts[0].data_type == ov::element::f16 && params.output_layouts[0].data_type == ov::element::f16;
}

// GQA-packed MIXED micro SDPA for small per-sequence q_len (speculative decoding verify): one work-group computes all
// query heads of a KV head for wg_tile_n / group tokens, so the KV cache is read once per KV head instead of once per
// query head, and the query tile is not padded with unused rows. Used if every subsequence has q_len <= this value.
constexpr size_t pa_micro_gqa_packed_max_q = 16;

// Causal full attention with grouped KV heads only (the packed columns share one causal/window rule per token).
inline bool pa_micro_gqa_packed_supported(const paged_attention& desc) {
    return desc.heads_num > desc.kv_heads_num && desc.sliding_window == 0 && !desc.has_sink_input && !desc.has_token_type_ids &&
           !desc.has_qq_bias && !desc.has_scores_output() && !desc.has_adaptive_rkv;
}

struct PagedAttentionOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::paged_attention::opt")
    explicit PagedAttentionOpt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_q_types = {
            ov::element::f32,
            ov::element::f16,
        };
        static constexpr std::array supported_kv_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i8,
        };

        auto desc = node.as<paged_attention>().get_primitive();
        if (desc->has_xattention) {
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false because XAttention is not supported with ocl. " << std::endl;
            return false;
        }

        // OCL PA kernel requires head_size >= 16 (SUBGROUPS_PER_WG = head_size / SUBGROUP_SIZE)
        OPENVINO_ASSERT(desc->k_head_size >= 16 && desc->v_head_size >= 16,
                        "[GPU] OCL PagedAttention kernel does not support head_size < 16."
                        " k_head_size=",
                        desc->k_head_size,
                        ", v_head_size=",
                        desc->v_head_size);

        const auto& q_layout = node.get_input_layout(0);
        const auto& k_layout = node.get_input_layout(1);
        const auto& v_layout = node.get_input_layout(2);
        const auto& out_layout = node.get_output_layout(0);
        if (!everyone_is(format::bfyx, q_layout.format, k_layout.format, v_layout.format, out_layout.format)) {
            return false;
        }

        if (!one_of(k_layout.data_type, supported_kv_types) || !one_of(v_layout.data_type, supported_kv_types)) {
            return false;
        }

        return one_of(q_layout.data_type, supported_q_types) && one_of(out_layout.data_type, supported_q_types);
    }
};

}  // namespace ov::intel_gpu::ocl
